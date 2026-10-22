"""Thin FFmpeg / audio helpers."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def ffmpeg(*args, quiet=True):
    if not FFMPEG:
        raise RuntimeError("FFmpeg was not found on PATH. Install it (winget install Gyan.FFmpeg) and retry.")
    command = [FFMPEG, "-hide_banner", "-y"]
    if quiet:
        command += ["-loglevel", "error"]
    command += [str(arg) for arg in args]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {' '.join(command)}\n{result.stderr.strip()}")


def probe(path):
    if not FFPROBE:
        raise RuntimeError("ffprobe was not found on PATH.")
    result = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration:stream=codec_type,r_frame_rate",
         "-of", "json", str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True,
    )
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    fps = None
    for stream in streams:
        if stream.get("codec_type") == "video":
            num, _, den = stream.get("r_frame_rate", "0/1").partition("/")
            fps = float(num) / float(den or 1) if float(den or 1) else None
    return {
        "duration": float(data["format"]["duration"]),
        "has_video": any(s.get("codec_type") == "video" for s in streams),
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
        "fps": fps,
    }


def extract_audio(video_path, wav_path, sample_rate=44100, channels=2):
    ffmpeg("-i", video_path, "-vn", "-ac", channels, "-ar", sample_rate, "-c:a", "pcm_s16le", wav_path)
    return Path(wav_path)


def load_mono(path, sample_rate):
    """Load any audio file as float32 mono at the requested sample rate."""
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != sample_rate:
        import librosa

        audio = librosa.resample(audio, orig_sr=rate, target_sr=sample_rate)
    return audio.astype(np.float32)


def time_stretch(audio, sample_rate, tempo, work_dir):
    """Speed audio up (tempo > 1) or down without changing pitch."""
    if abs(tempo - 1.0) < 0.01:
        return audio
    work_dir = Path(work_dir)
    source, target = work_dir / "_stretch_in.wav", work_dir / "_stretch_out.wav"
    sf.write(source, audio, sample_rate)
    try:
        ffmpeg("-i", source, "-filter:a", f"rubberband=tempo={tempo:.5f}:pitchq=quality", target)
    except RuntimeError:
        # FFmpeg builds without librubberband: chain atempo (each stage limited to 0.5..2.0).
        stages, remaining = [], tempo
        while remaining > 2.0:
            stages.append(2.0)
            remaining /= 2.0
        while remaining < 0.5:
            stages.append(0.5)
            remaining /= 0.5
        stages.append(remaining)
        ffmpeg("-i", source, "-filter:a", ",".join(f"atempo={s:.5f}" for s in stages), target)
    stretched, _ = sf.read(str(target), dtype="float32")
    return stretched


def trim_silence(audio, sample_rate, threshold_db=-40.0):
    """Strip leading/trailing silence that TTS engines pad around speech."""
    if audio.size == 0:
        return audio
    frame = max(1, int(sample_rate * 0.01))
    peak = float(np.max(np.abs(audio))) or 1.0
    threshold = peak * (10 ** (threshold_db / 20))
    loud = np.flatnonzero(np.abs(audio) > threshold)
    if loud.size == 0:
        return audio
    start = max(0, loud[0] - frame)
    end = min(audio.size, loud[-1] + frame)
    return audio[start:end]
