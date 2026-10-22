"""Approximate lip sync with Wav2Lip (runs on CPU).

Frames are streamed through the model in batches and piped straight into FFmpeg,
so memory use stays flat even for long 1080p videos. Only frames where someone is
speaking (originally or in the dub) are re-rendered; everything else is copied.
"""

import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from . import media, models

IMG_SIZE = 96
MEL_STEP = 16
BATCH_SIZE = 32


# --- Wav2Lip network (architecture from github.com/Rudrabha/Wav2Lip) -------------

class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False):
        super().__init__()
        self.conv_block = nn.Sequential(nn.Conv2d(cin, cout, kernel_size, stride, padding), nn.BatchNorm2d(cout))
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out = out + x
        return self.act(out)


class Conv2dTranspose(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding), nn.BatchNorm2d(cout)
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.conv_block(x))


def _res(channels, count):
    return [Conv2d(channels, channels, 3, 1, 1, residual=True) for _ in range(count)]


class Wav2Lip(nn.Module):
    def __init__(self):
        super().__init__()
        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(6, 16, 7, 1, 3)),
            nn.Sequential(Conv2d(16, 32, 3, 2, 1), *_res(32, 2)),
            nn.Sequential(Conv2d(32, 64, 3, 2, 1), *_res(64, 3)),
            nn.Sequential(Conv2d(64, 128, 3, 2, 1), *_res(128, 2)),
            nn.Sequential(Conv2d(128, 256, 3, 2, 1), *_res(256, 2)),
            nn.Sequential(Conv2d(256, 512, 3, 2, 1), *_res(512, 1)),
            nn.Sequential(Conv2d(512, 512, 3, 1, 0), Conv2d(512, 512, 1, 1, 0)),
        ])
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, 3, 1, 1), *_res(32, 2),
            Conv2d(32, 64, 3, (3, 1), 1), *_res(64, 2),
            Conv2d(64, 128, 3, 3, 1), *_res(128, 2),
            Conv2d(128, 256, 3, (3, 2), 1), *_res(256, 1),
            Conv2d(256, 512, 3, 1, 0),
            Conv2d(512, 512, 1, 1, 0),
        )
        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(512, 512, 1, 1, 0)),
            nn.Sequential(Conv2dTranspose(1024, 512, 3, 1, 0), *_res(512, 1)),
            nn.Sequential(Conv2dTranspose(1024, 512, 3, 2, 1, output_padding=1), *_res(512, 2)),
            nn.Sequential(Conv2dTranspose(768, 384, 3, 2, 1, output_padding=1), *_res(384, 2)),
            nn.Sequential(Conv2dTranspose(512, 256, 3, 2, 1, output_padding=1), *_res(256, 2)),
            nn.Sequential(Conv2dTranspose(320, 128, 3, 2, 1, output_padding=1), *_res(128, 2)),
            nn.Sequential(Conv2dTranspose(160, 64, 3, 2, 1, output_padding=1), *_res(64, 2)),
        ])
        self.output_block = nn.Sequential(Conv2d(80, 32, 3, 1, 1), nn.Conv2d(32, 3, 1, 1, 0), nn.Sigmoid())

    def forward(self, audio, faces):
        x = self.audio_encoder(audio)
        feats = []
        y = faces
        for block in self.face_encoder_blocks:
            y = block(y)
            feats.append(y)
        for block in self.face_decoder_blocks:
            x = block(x)
            x = torch.cat((x, feats.pop()), dim=1)
        return self.output_block(x)


class SyncNet(nn.Module):
    """Wav2Lip's lip-sync expert: embeds 5 mouth frames and 0.2 s of audio into one space.

    High cosine similarity at the true time offset (and low at wrong offsets) means the
    visible face is the one producing the sound.
    """

    def __init__(self):
        super().__init__()
        self.face_encoder = nn.Sequential(
            Conv2d(15, 32, (7, 7), 1, 3),
            Conv2d(32, 64, 5, (1, 2), 1), *_res(64, 2),
            Conv2d(64, 128, 3, 2, 1), *_res(128, 3),
            Conv2d(128, 256, 3, 2, 1), *_res(256, 2),
            Conv2d(256, 512, 3, 2, 1), *_res(512, 2),
            Conv2d(512, 512, 3, 2, 1),
            Conv2d(512, 512, 3, 1, 0),
            Conv2d(512, 512, 1, 1, 0),
        )
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, 3, 1, 1), *_res(32, 2),
            Conv2d(32, 64, 3, (3, 1), 1), *_res(64, 2),
            Conv2d(64, 128, 3, 3, 1), *_res(128, 2),
            Conv2d(128, 256, 3, (3, 2), 1), *_res(256, 2),
            Conv2d(256, 512, 3, 1, 0),
            Conv2d(512, 512, 1, 1, 0),
        )

    def forward(self, audio, faces):
        audio = nn.functional.normalize(self.audio_encoder(audio).flatten(1), dim=1)
        faces = nn.functional.normalize(self.face_encoder(faces).flatten(1), dim=1)
        return audio, faces


def load_syncnet():
    checkpoint = torch.load(models.syncnet_checkpoint(), map_location="cpu", weights_only=False)
    state = {key.replace("module.", "", 1): value for key, value in checkpoint["state_dict"].items()}
    model = SyncNet()
    model.load_state_dict(state)
    return model.eval()


def load_model():
    checkpoint = torch.load(models.wav2lip_checkpoint(), map_location="cpu", weights_only=False)
    state = {key.replace("module.", "", 1): value for key, value in checkpoint["state_dict"].items()}
    model = Wav2Lip()
    model.load_state_dict(state)
    return model.eval()


# --- Audio features (matches Wav2Lip's hparams) --------------------------------------

def melspectrogram(wav):
    import librosa
    from scipy.signal import lfilter

    wav = lfilter([1, -0.97], [1], wav)
    spectrum = np.abs(librosa.stft(y=wav, n_fft=800, hop_length=200, win_length=800))
    mel_basis = librosa.filters.mel(sr=16000, n_fft=800, n_mels=80, fmin=55, fmax=7600)
    mel_db = 20 * np.log10(np.maximum(1e-5, mel_basis @ spectrum)) - 20
    return np.clip(8 * ((mel_db + 100) / 100) - 4, -4, 4).astype(np.float32)


def mel_chunk(mel, frame_index, fps):
    start = int(frame_index * 80.0 / fps)
    if start + MEL_STEP > mel.shape[1]:
        return mel[:, -MEL_STEP:]
    return mel[:, start:start + MEL_STEP]


# --- Face tracking -------------------------------------------------------------------

# Each tracked frame is a row of 14 numbers: the Wav2Lip crop box (x1, y1, x2, y2)
# followed by five landmarks in image order: left eye, right eye, nose, left and
# right mouth corner (x, y each).
TRACK_COLUMNS = 14


class FaceDetector:
    """YuNet face + landmark detector, run on a downscaled frame for speed."""

    def __init__(self, width, height):
        self.scale = min(1.0, 640.0 / max(width, height))
        self.size = (int(width * self.scale), int(height * self.scale))
        self.detector = cv2.FaceDetectorYN.create(str(models.yunet_model()), "", self.size, 0.6, 0.3, 5)

    def landmarks(self, frame):
        """Return the five landmarks (5x2, image left-to-right order) of the largest face, or None."""
        small = cv2.resize(frame, self.size) if self.scale < 1.0 else frame
        _, faces = self.detector.detect(small)
        if faces is None or len(faces) == 0:
            return None
        face = faces[np.argmax(faces[:, 2] * faces[:, 3])][4:14].reshape(5, 2) / self.scale
        eyes = face[np.argsort(face[:2, 0])]
        mouth = face[3:5][np.argsort(face[3:5, 0])]
        return np.vstack([eyes, face[2:3], mouth])

    def __call__(self, frame):
        """Return the Wav2Lip crop box [x1, y1, x2, y2] of the largest face, or None."""
        points = self.landmarks(frame)
        return None if points is None else crop_box(points)


def crop_box(points):
    eyes, mouth = points[:2].mean(axis=0), points[3:5].mean(axis=0)
    d = mouth[1] - eyes[1]            # eye-to-mouth distance drives the crop size
    if d < 6:
        return None
    # Wav2Lip was trained on S3FD crops: eyes ~36% and mouth ~71% down a near-square box
    # ending just under the chin. Placing the mouth where the model expects it is what
    # keeps the generated lips on the real lips instead of drifting toward the nose.
    height = 2.8 * d
    width = 0.95 * height
    center_x = (eyes[0] + mouth[0]) / 2
    top = eyes[1] - 1.0 * d
    return np.array([center_x - width / 2, top, center_x + width / 2, top + height])


def detect_faces(video_path, frame_count, width, height):
    """One pass over the video: face box + landmarks per frame (NaN where no face)."""
    detector = FaceDetector(width, height)
    tracks = np.full((frame_count, TRACK_COLUMNS), np.nan, dtype=np.float32)
    capture = cv2.VideoCapture(str(video_path))
    for index in range(frame_count):
        ok, frame = capture.read()
        if not ok:
            break
        points = detector.landmarks(frame)
        box = None if points is None else crop_box(points)
        if box is not None:
            tracks[index] = np.concatenate([box, points.ravel()])
    capture.release()
    return tracks


def smooth_tracks(tracks, width, height):
    """Bridge short drop-outs and smooth with a centred window (no lag when the head moves)."""
    found = ~np.isnan(tracks[:, 0])
    if not found.any():
        return tracks
    frames = np.arange(len(tracks))
    known = np.flatnonzero(found)
    gap = np.abs(frames[:, None] - known[None, :]).min(axis=1)
    filled = tracks.copy()
    for column in range(TRACK_COLUMNS):
        filled[:, column] = np.where(gap <= 6, np.interp(frames, known, tracks[known, column]), np.nan)
    window = 5
    smoothed = filled.copy()
    for index in range(len(filled)):
        if not np.isnan(filled[index, 0]):
            smoothed[index] = np.nanmean(filled[max(0, index - window // 2):index + window // 2 + 1], axis=0)
    smoothed[:, [0, 2]] = np.clip(smoothed[:, [0, 2]], 0, width)
    smoothed[:, [1, 3]] = np.clip(smoothed[:, [1, 3]], 0, height)
    return smoothed


SYNC_MIN_CONFIDENCE = 0.065   # real on-screen speech scored 0.08-0.21, voice-over 0.01-0.05
SYNC_MAX_OFFSET = 4           # frames; voice-over has no consistent best offset
SYNC_MIN_FRAMES = 12
MIN_FACE_SHARE = 0.12         # face crop height vs frame height; tiny background faces are skipped


def _face_runs(tracks, start, end):
    """Split [start, end) into continuous shots of one face (breaks on gaps and hard cuts)."""
    runs, run_start = [], None
    for index in range(start, end + 1):
        present = index < end and not np.isnan(tracks[index, 0])
        if present and run_start is not None:
            previous, current = tracks[index - 1, :4], tracks[index, :4]
            size = previous[2] - previous[0]
            jump = np.abs((previous[:2] + previous[2:]) / 2 - (current[:2] + current[2:]) / 2).max()
            if jump > 0.3 * size or abs((current[2] - current[0]) / size - 1) > 0.25:
                runs.append((run_start, index))
                run_start = index
        if present and run_start is None:
            run_start = index
        if not present and run_start is not None:
            runs.append((run_start, index))
            run_start = None
    return runs


def on_screen_speaker(video_path, tracks, original_audio, segments, fps, log=print):
    """Return a per-frame mask of where the visible face is really speaking the original line.

    Every continuous face shot inside a spoken line is checked with SyncNet against the
    *original* audio. Voice-over on top of a silent face, narrators, stock footage and
    off-screen speakers do not line up with the lips, so those frames are left untouched.
    """
    count = len(tracks)
    frame_height = float(np.nanmax(tracks[:, 3])) if np.isfinite(tracks[:, 3]).any() else 0.0
    capture = cv2.VideoCapture(str(video_path))
    frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or frame_height
    capture.release()
    wanted = np.zeros(count, dtype=bool)
    candidate_runs = []
    for segment in segments:
        start, end = max(0, int(segment.start * fps)), min(count, int(np.ceil(segment.end * fps)))
        for run in _face_runs(tracks, start, end):
            face_height = float(np.median(tracks[run[0]:run[1], 3] - tracks[run[0]:run[1], 1]))
            if run[1] - run[0] >= SYNC_MIN_FRAMES and face_height >= MIN_FACE_SHARE * frame_height:
                candidate_runs.append(run)
                wanted[run[0]:run[1]] = True
    speaking = np.zeros(count, dtype=bool)
    if not candidate_runs:
        log("  No visible face during speech: the whole dub is treated as voice-over.")
        return speaking

    # Collect the lower-half face crops SyncNet needs (only for candidate frames).
    lows = {}
    capture = cv2.VideoCapture(str(video_path))
    for index in range(count):
        ok, frame = capture.read()
        if not ok:
            break
        if wanted[index]:
            x1, y1, x2, y2 = tracks[index, :4].astype(int)
            face = frame[max(0, y1):y2, max(0, x1):x2]
            if face.size:
                lows[index] = cv2.resize(face, (IMG_SIZE, IMG_SIZE))[IMG_SIZE // 2:]
    capture.release()

    net = load_syncnet()
    mel = melspectrogram(media.load_mono(original_audio, 16000))
    window = range(-15, 16)
    scored = []
    for start, end in candidate_runs:
        times = [t for t in range(start, end - 4) if all(i in lows for i in range(t, t + 5))]
        if len(times) < 4:
            continue
        faces = np.stack([np.concatenate([lows[i] for i in range(t, t + 5)], axis=2) for t in times])
        audio_times = sorted({t + o for t in times for o in window if 0 <= t + o < count})
        with torch.inference_mode():
            _, face_emb = net(torch.zeros(1, 1, 80, MEL_STEP),
                              torch.from_numpy(faces.astype(np.float32) / 255.0).permute(0, 3, 1, 2))
            mels = np.stack([mel_chunk(mel, t, fps) for t in audio_times])[:, None]
            audio_emb, _ = net(torch.from_numpy(mels), torch.zeros(1, 15, IMG_SIZE // 2, IMG_SIZE))
        audio_index = {t: k for k, t in enumerate(audio_times)}
        face_emb, audio_emb = face_emb.numpy(), audio_emb.numpy()
        similarity = {}
        for offset in window:
            pairs = [(k, audio_index[t + offset]) for k, t in enumerate(times) if t + offset in audio_index]
            if pairs:
                similarity[offset] = float(np.mean([face_emb[k] @ audio_emb[j] for k, j in pairs]))
        best = max(similarity, key=similarity.get)
        scored.append((start, end, similarity, best, similarity[best] - float(np.median(list(similarity.values())))))

    # Many recordings have a constant audio/picture delay (often 3-6 frames). Estimate it from
    # the clearest shots and judge every shot relative to it; voice-over has no stable offset.
    confident = [best for *_, best, confidence in scored if confidence >= 0.1 and abs(best) <= 10]
    av_offset = int(np.median(confident)) if confident else 0
    if av_offset:
        log(f"  Source audio/picture offset: {av_offset:+d} frames")
    for start, end, similarity, best, _ in scored:
        near = [similarity[o] for o in range(av_offset - SYNC_MAX_OFFSET, av_offset + SYNC_MAX_OFFSET + 1)
                if o in similarity]
        confidence = max(near) - float(np.median(list(similarity.values())))
        on_screen = confidence >= SYNC_MIN_CONFIDENCE and abs(best - av_offset) <= SYNC_MAX_OFFSET
        if on_screen:
            speaking[start:end] = True
        log(f"  {start / fps:6.2f}-{end / fps:6.2f}s face shot: sync {confidence:.3f} at {best:+d} frames -> "
            f"{'speaker on screen, lip sync' if on_screen else 'voice-over / not speaking, keep original'}")
    return speaking


def speech_frames(speaking, tracks, segments, voice_track, voice_rate, fps):
    """Frames to re-render: confirmed on-screen speech, extended while the dub is still talking.

    The dubbed line can run a little longer than the original; we follow it while the same
    face stays visible, but never into the next line.
    """
    count = len(speaking)
    hop = int(voice_rate / fps)
    energy = np.zeros(count, dtype=np.float32)
    if hop:
        usable = min(count, voice_track.size // hop)
        energy[:usable] = np.sqrt(np.mean(voice_track[: usable * hop].reshape(usable, hop) ** 2, axis=1))
    loud = energy > max(1e-3, energy.max() * 0.02)
    starts = sorted(int(s.start * fps) for s in segments)
    active = speaking.copy()
    for edge in np.flatnonzero(speaking[:-1] & ~speaking[1:]) + 1:
        limit = next((s for s in starts if s >= edge), count)
        tail = edge
        while tail < limit and loud[tail] and not np.isnan(tracks[tail, 0]):
            tail += 1
        active[edge:tail] = True
    return active


# --- Compositing -----------------------------------------------------------------------

# FFHQ 5-point template (GFPGAN's alignment) for a 512x512 face.
FFHQ_512 = np.array([[0.37691676, 0.46864664], [0.62285697, 0.46912813], [0.50123859, 0.61331904],
                     [0.39308822, 0.72541100], [0.61150205, 0.72490465]], dtype=np.float32) * 512


class FaceEnhancer:
    """GFPGAN v1.4 (Apache-2.0) via ONNX Runtime: restores detail lost by Wav2Lip's 96 px output."""

    def __init__(self):
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = max(1, (os.cpu_count() or 8) // 2)
        self.session = onnxruntime.InferenceSession(str(models.gfpgan_model()), options,
                                                    providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, frame, points):
        """Return (restored frame-sized image, validity mask) for the face at `points`."""
        matrix = cv2.estimateAffinePartial2D(points, FFHQ_512, method=cv2.LMEDS)[0]
        aligned = cv2.warpAffine(frame, matrix, (512, 512), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        blob = (cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)[None]
        restored = self.session.run(None, {self.input_name: blob})[0][0].transpose(1, 2, 0)
        restored = cv2.cvtColor(np.clip((restored + 1.0) * 127.5, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        inverse = cv2.invertAffineTransform(matrix)
        size = (frame.shape[1], frame.shape[0])
        back = cv2.warpAffine(restored, inverse, size, flags=cv2.INTER_LINEAR)
        valid = cv2.warpAffine(np.ones((512, 512), np.float32), inverse, size, flags=cv2.INTER_LINEAR)
        return back, valid


def mouth_mask(shape, points):
    """Soft ellipse over the lips and the jaw that moves with them (not the whole lower face)."""
    eyes, mouth = points[:2].mean(axis=0), points[3:5].mean(axis=0)
    d = max(mouth[1] - eyes[1], 6.0)
    mouth_width = np.linalg.norm(points[4] - points[3])
    mask = np.zeros(shape[:2], dtype=np.float32)
    center = (int(mouth[0]), int(mouth[1] + 0.18 * d))
    axes = (int(0.95 * mouth_width), int(0.62 * d))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    sigma = max(2.0, 0.16 * d)
    return cv2.GaussianBlur(mask, (0, 0), sigma)[..., None]


def match_color(source, target, mask):
    """Shift `source`'s colour (LAB mean/std inside mask) to match `target`, removing the dark 'shade'."""
    # Compare the feathered border only: it is mostly skin in both images, whereas the
    # centre holds the mouth itself, which is *supposed* to differ (open vs closed).
    weights = (mask[..., 0] > 0.15) & (mask[..., 0] < 0.85)
    if weights.sum() < 20:
        return source
    src = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    dst = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    s_mean, s_std = src[weights].mean(axis=0), src[weights].std(axis=0) + 1e-3
    d_mean, d_std = dst[weights].mean(axis=0), dst[weights].std(axis=0) + 1e-3
    scale = np.clip(d_std / s_std, 0.85, 1.15)
    out = (src - s_mean) * scale + d_mean
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


# --- Main entry point ----------------------------------------------------------------

def lip_sync(video_path, audio_path, out_path, segments, voice_track, voice_rate, work_dir,
             original_audio=None, enhance=True, log=print):
    torch.set_num_threads(max(1, (os.cpu_count() or 8) // 2))
    work_dir = Path(work_dir)
    info = media.probe(video_path)
    fps = min(round(info["fps"] or 25), 30)

    # Constant frame rate source so frame N lines up with audio time N / fps.
    cfr_path = work_dir / "lipsync_cfr.mp4"
    log(f"Preparing {fps} fps video for lip sync...")
    media.ffmpeg("-i", video_path, "-an", "-vf", f"fps={fps}", "-c:v", "libx264", "-preset", "veryfast",
                 "-crf", "16", "-pix_fmt", "yuv420p", cfr_path)

    capture = cv2.VideoCapture(str(cfr_path))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or int(info["duration"] * fps)

    log("Loading Wav2Lip model...")
    model = load_model()
    # Drive the mouth from the dubbed voice only: background music confuses Wav2Lip.
    import librosa

    mel = melspectrogram(librosa.resample(np.asarray(voice_track, dtype=np.float32), orig_sr=voice_rate, target_sr=16000))
    log("Tracking faces and checking who is speaking on screen (SyncNet)...")
    raw_tracks = detect_faces(cfr_path, frame_count, width, height)
    speaking = on_screen_speaker(cfr_path, raw_tracks, original_audio or audio_path, segments, fps, log=log)
    active = speech_frames(speaking, raw_tracks, segments, voice_track, voice_rate, fps)
    tracks = smooth_tracks(raw_tracks, width, height)
    tracks[~active] = np.nan
    log(f"Lip sync will re-render {int((active & ~np.isnan(tracks[:, 0])).sum())} of {frame_count} frames.")
    enhancer = None
    if enhance:
        try:
            log("Loading GFPGAN face enhancer...")
            enhancer = FaceEnhancer()
        except Exception as exc:  # still produce a lip-synced video, just softer
            log(f"WARNING: face enhancer unavailable ({exc}); continuing without it.")

    encoder = subprocess.Popen(
        [media.FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
         "-i", str(audio_path), "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-shortest", str(out_path)],
        stdin=subprocess.PIPE,
    )

    pending = []  # (frame, track row or None, frame_index)
    synced = 0

    def flush():
        nonlocal synced
        jobs = [(i, track) for i, (_, track, _) in enumerate(pending) if track is not None]
        if jobs:
            faces, mels = [], []
            for i, track in jobs:
                frame, _, frame_index = pending[i]
                x1, y1, x2, y2 = track[:4].astype(int)
                faces.append(cv2.resize(frame[y1:y2, x1:x2], (IMG_SIZE, IMG_SIZE)))
                mels.append(mel_chunk(mel, frame_index, fps))
            faces = np.asarray(faces, dtype=np.float32)
            masked = faces.copy()
            masked[:, IMG_SIZE // 2:] = 0
            face_batch = torch.from_numpy(np.concatenate((masked, faces), axis=3) / 255.0).permute(0, 3, 1, 2).float()
            mel_batch = torch.from_numpy(np.asarray(mels)[:, None])
            with torch.inference_mode():
                predictions = model(mel_batch, face_batch).permute(0, 2, 3, 1).numpy() * 255.0
            for (i, track), prediction in zip(jobs, predictions):
                frame = pending[i][0]
                x1, y1, x2, y2 = track[:4].astype(int)
                points = track[4:].reshape(5, 2)
                synthetic = frame.copy()
                synthetic[y1:y2, x1:x2] = cv2.resize(
                    np.clip(prediction, 0, 255).astype(np.uint8), (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)
                mask = mouth_mask(frame.shape, points)
                if enhancer is not None:
                    restored, valid = enhancer(synthetic, points)
                    mask = mask * valid[..., None]
                    synthetic = restored
                synthetic = match_color(synthetic, frame, mask)
                # Only the lips and moving jaw are replaced; beard, cheeks and skin stay original.
                ys, xs = np.nonzero(mask[..., 0] > 0.003)
                if ys.size:
                    t, b, l, r = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                    m = mask[t:b, l:r]
                    frame[t:b, l:r] = np.clip(m * synthetic[t:b, l:r] + (1 - m) * frame[t:b, l:r], 0, 255).astype(np.uint8)
            synced += len(jobs)
        for frame, _, _ in pending:
            encoder.stdin.write(frame.tobytes())
        pending.clear()

    try:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            track = tracks[index] if index < len(tracks) else None
            if track is not None and (np.isnan(track[0]) or track[2] - track[0] < 24 or track[3] - track[1] < 24):
                track = None
            pending.append((frame, track, index))
            if len(pending) >= BATCH_SIZE:
                flush()
            index += 1
            if index % (fps * 5) == 0:
                log(f"  lip sync: {index}/{frame_count} frames ({100 * index // max(frame_count, 1)}%)")
        flush()
    finally:
        capture.release()
        encoder.stdin.close()
        encoder.wait()
    if encoder.returncode != 0:
        raise RuntimeError("FFmpeg failed while writing the lip-synced video.")
    log(f"Lip sync finished: {synced} of {index} frames re-rendered.")
    return Path(out_path)
