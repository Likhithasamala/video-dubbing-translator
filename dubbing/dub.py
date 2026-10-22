"""Turn translated segments into a dubbed voice track aligned to the original timing."""

import numpy as np
import soundfile as sf

from . import media

SAMPLE_RATE = 48000
MAX_SPEEDUP = 1.35   # beyond this speech sounds rushed, so we borrow the pause after the line
MAX_STRETCH = 2.0    # hard limit before a line is cut off
FADE_SECONDS = 0.01


def detect_gender(vocals_path, segments):
    """Guess the main speaker's gender from median pitch (male < ~165 Hz < female)."""
    import librosa

    audio = media.load_mono(vocals_path, 16000)
    pieces = [audio[int(s.start * 16000):int(s.end * 16000)] for s in segments[:40]]
    pieces = [p for p in pieces if p.size > 1600]
    if not pieces:
        return "male", None
    speech = np.concatenate(pieces)[: 16000 * 90]
    f0 = librosa.yin(speech, fmin=65, fmax=400, sr=16000, frame_length=1024)
    rms = librosa.feature.rms(y=speech, frame_length=1024, hop_length=256)[0][: f0.size]
    voiced = f0[: rms.size][rms > np.percentile(rms, 50)]
    voiced = voiced[(voiced > 70) & (voiced < 380)]
    if voiced.size == 0:
        return "male", None
    pitch = float(np.median(voiced))
    return ("female" if pitch > 165 else "male"), pitch


def _fit(engine, text, slot, window, work_dir, synth_kwargs):
    """Synthesize text and squeeze it into the slot (or the wider window if it must)."""
    audio, rate = engine.synthesize(text, speed=1.0, **synth_kwargs)
    audio = media.trim_silence(audio, rate)
    length = audio.size / rate

    # Engines with a native speaking-rate control sound better than post-stretching.
    if length > slot * 1.05 and engine.name in ("piper", "mms"):
        speed = min(length / slot, 1.25)
        audio, rate = engine.synthesize(text, speed=speed, **synth_kwargs)
        audio = media.trim_silence(audio, rate)
        length = audio.size / rate

    if rate != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=rate, target_sr=SAMPLE_RATE)
    length = audio.size / SAMPLE_RATE

    if length > slot:
        target = slot if length / slot <= MAX_SPEEDUP else min(window, max(slot, length / MAX_SPEEDUP))
        tempo = min(length / target, MAX_STRETCH)
    elif length < slot * 0.8:
        tempo = max(length / slot, 0.88)   # slightly slower speech fills long lines more naturally
    else:
        tempo = 1.0
    audio = media.time_stretch(audio, SAMPLE_RATE, tempo, work_dir)
    return audio, tempo


def render_voice_track(segments, engine, total_duration, work_dir, reference_clips=None, log=print):
    """Return a float32 mono voice track at SAMPLE_RATE covering the whole video."""
    track = np.zeros(int(total_duration * SAMPLE_RATE) + SAMPLE_RATE, dtype=np.float32)
    fade = int(FADE_SECONDS * SAMPLE_RATE)
    for index, segment in enumerate(segments):
        text = segment.translation.strip()
        if not text:
            continue
        next_start = segments[index + 1].start if index + 1 < len(segments) else total_duration
        slot = max(segment.duration, 0.3)
        window = max(next_start - segment.start - 0.05, slot)
        synth_kwargs = {}
        if reference_clips:
            synth_kwargs = {"reference_wav": reference_clips[index], "prompt_text": segment.text, "seed": 42 + index}

        audio, tempo = _fit(engine, text, slot, window, work_dir, synth_kwargs)
        max_samples = int(window * SAMPLE_RATE)
        if audio.size > max_samples:
            log(f"  ! line {index + 1} is {audio.size / SAMPLE_RATE:.2f}s for a {window:.2f}s gap; trimming the tail")
            audio = audio[:max_samples]
        if audio.size > 2 * fade:
            ramp = np.linspace(0, 1, fade, dtype=np.float32)
            audio[:fade] *= ramp
            audio[-fade:] *= ramp[::-1]

        start = int(segment.start * SAMPLE_RATE)
        track[start:start + audio.size] += audio[: track.size - start]
        log(f"  {index + 1}/{len(segments)} [{segment.start:6.2f}s] x{tempo:.2f} {text}")
    return track[: int(total_duration * SAMPLE_RATE)]


def _rms(audio):
    return float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0


def mix(voice, vocals_path, background_path, segments, out_wav):
    """Mix the dubbed voice over the background, matching the original voice loudness."""
    if vocals_path is not None:
        original = media.load_mono(vocals_path, SAMPLE_RATE)
        spoken = [original[int(s.start * SAMPLE_RATE):int(s.end * SAMPLE_RATE)] for s in segments]
        reference_rms = _rms(np.concatenate(spoken)) if spoken else 0.0
    else:
        reference_rms = 0.0
    voice_rms = _rms(voice[np.abs(voice) > 1e-4])
    if reference_rms > 0 and voice_rms > 0:
        voice = voice * min(reference_rms / voice_rms, 4.0)
    else:
        voice = voice * (0.25 / max(voice_rms, 1e-6))

    if background_path is not None:
        background, rate = sf.read(str(background_path), dtype="float32", always_2d=True)
        if rate != SAMPLE_RATE:
            import librosa

            background = librosa.resample(background.T, orig_sr=rate, target_sr=SAMPLE_RATE).T
    else:
        background = np.zeros((voice.size, 2), dtype=np.float32)

    if background.shape[1] == 1:
        background = np.repeat(background, 2, axis=1)
    length = voice.size
    if background.shape[0] < length:
        background = np.pad(background, ((0, length - background.shape[0]), (0, 0)))
    stereo = background[:length] + voice[:, None]
    peak = float(np.max(np.abs(stereo))) if stereo.size else 0.0
    if peak > 0.98:
        stereo *= 0.98 / peak
    sf.write(str(out_wav), stereo, SAMPLE_RATE, subtype="PCM_16")
    return voice
