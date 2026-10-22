"""End-to-end dubbing pipeline.

    video ─► extract audio ─► split voice / background ─► Whisper transcript
          ─► NLLB translation ─► TTS per line, time-fitted ─► mix with background
          ─► dubbed video (+ optional Wav2Lip lip sync), audio files and subtitles

Speech recognition and separation run once per video and are cached, so dubbing
into extra languages later only pays for translation + speech synthesis.
"""

import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from . import asr, dub, media
from .languages import BY_CODE, get_language

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = BASE_DIR / "outputs"
VOCAL_REMOVER_DIR = BASE_DIR / "vocal-remover"


def progress(percent, message):
    print(f"PROGRESS {int(percent)} {message}", flush=True)


def _log(message):
    print(message, flush=True)


def _safe_name(text):
    return re.sub(r"[^\w\-]+", "_", text, flags=re.UNICODE).strip("_")[:60] or "video"


def _fingerprint(path):
    stat = Path(path).stat()
    return hashlib.sha1(f"{Path(path).resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()[:10]


def _srt_time(seconds):
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def write_srt(segments, path, field="translation"):
    lines = []
    for index, segment in enumerate(segments, 1):
        text = getattr(segment, field).strip()
        if text:
            lines += [str(index), f"{_srt_time(segment.start)} --> {_srt_time(segment.end)}", text, ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def separate_vocals(audio_path, work_dir, log=_log):
    """Split speech from music/effects with the bundled vocal-remover model."""
    audio_path, work_dir = Path(audio_path).resolve(), Path(work_dir).resolve()
    vocals = work_dir / f"{audio_path.stem}_Vocals.wav"
    background = work_dir / f"{audio_path.stem}_Instruments.wav"
    if vocals.exists() and background.exists():
        return vocals, background
    model = VOCAL_REMOVER_DIR / "models" / "baseline.pth"
    if not model.exists():
        raise FileNotFoundError(f"Vocal-remover model missing: {model}. Run: python scripts/setup_models.py")
    result = subprocess.run(
        [sys.executable, "inference.py", "--input", str(audio_path), "--output_dir", str(work_dir) + "/"],
        cwd=VOCAL_REMOVER_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace",
    )
    if result.returncode != 0 or not vocals.exists():
        raise RuntimeError(f"Vocal separation failed:\n{result.stdout[-2000:]}")
    return vocals, background


def mux_video(video_path, audio_path, out_path):
    try:
        media.ffmpeg("-i", video_path, "-i", audio_path, "-map", "0:v:0", "-map", "1:a:0",
                     "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out_path)
    except RuntimeError:
        # Source codec not allowed in MP4 (e.g. some WebM/MKV files): re-encode the picture.
        media.ffmpeg("-i", video_path, "-i", audio_path, "-map", "0:v:0", "-map", "1:a:0",
                     "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                     "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out_path)


def _reference_clips(vocals_path, segments, folder):
    folder.mkdir(parents=True, exist_ok=True)
    audio = media.load_mono(vocals_path, 16000)
    clips = []
    for index, segment in enumerate(segments):
        path = folder / f"{index + 1}.wav"
        sf.write(path, audio[int(segment.start * 16000):int(segment.end * 16000)], 16000)
        clips.append(path)
    return clips


def run(
    video,
    targets,
    source="auto",
    whisper_model="medium",
    use_indic=True,
    nllb_model="600M",
    tts_engine="auto",
    voice="auto",
    separate=True,
    lip_sync=False,
    output_root=OUTPUT_ROOT,
    log=_log,
):
    started = time.time()
    video = Path(video).expanduser().resolve()
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")
    info = media.probe(video)
    if not info["has_audio"]:
        raise RuntimeError("This video has no audio track, so there is nothing to translate.")
    target_languages = [get_language(code) for code in targets]
    source_language = get_language(source, allow_auto=True)

    job_dir = Path(output_root) / f"{_safe_name(video.stem)}_{_fingerprint(video)}"
    work_dir = job_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    log(f"Job folder: {job_dir}")

    # 1. Audio + separation -------------------------------------------------------
    progress(2, "Extracting audio")
    audio_path = work_dir / "original.wav"
    if not audio_path.exists():
        media.extract_audio(video, audio_path)

    vocals_path, background_path = audio_path, None
    if separate:
        progress(5, "Separating voice from background music")
        try:
            vocals_path, background_path = separate_vocals(audio_path, work_dir, log)
        except Exception as exc:  # keep going: dubbing without the music bed is still useful
            log(f"WARNING: {exc}\nContinuing without background music.")

    # 2. Transcription (cached) ---------------------------------------------------
    progress(20, "Transcribing speech")
    transcript_path = work_dir / f"transcript_{whisper_model}{'' if use_indic else '_generic'}_{source_language.code if source_language else 'auto'}.json"
    if transcript_path.exists():
        cached = json.loads(transcript_path.read_text(encoding="utf-8"))
        segments = [asr.Segment(s["start"], s["end"], s["text"]) for s in cached["segments"]]
        detected = cached["language"]
        log(f"Using cached transcript ({len(segments)} lines, language={detected}).")
    else:
        segments, detected = asr.transcribe(
            vocals_path, language=source_language.code if source_language else None,
            model_size=whisper_model, use_indic=use_indic, log=log,
        )
        transcript_path.write_text(json.dumps(
            {"language": detected, "segments": [s.to_dict() for s in segments]}, ensure_ascii=False, indent=2,
        ), encoding="utf-8")
    if not segments:
        raise RuntimeError("No speech was detected in this video.")
    source_language = source_language or BY_CODE.get(detected)
    if source_language is None:
        raise RuntimeError(f"Detected language '{detected}' is not supported yet. Pick the source language manually.")
    write_srt(segments, job_dir / f"subtitles_{source_language.code}_original.srt", field="text")

    gender = voice
    if voice == "auto":
        gender, pitch = dub.detect_gender(vocals_path, segments)
        log(f"Speaker voice: {gender}" + (f" (median pitch {pitch:.0f} Hz)" if pitch else ""))

    # 3. Speech -> English (cached) -------------------------------------------------
    # Whisper's translate task listens to the speech and writes English directly, which
    # handles Indian-language speech mixed with English words far better than
    # transcribing first and translating the text. Other targets then start from this
    # English, NLLB's strongest direction.
    english = BY_CODE["en"]
    if source_language.code == "en":
        english_segments = segments
        for segment in english_segments:
            segment.translation = segment.text
    else:
        progress(32, "Translating speech to English")
        english_path = work_dir / f"translate_{whisper_model}_{source_language.code}.json"
        if english_path.exists():
            cached = json.loads(english_path.read_text(encoding="utf-8"))
            english_segments = [asr.Segment(s["start"], s["end"], s["text"]) for s in cached["segments"]]
            log(f"Using cached English translation ({len(english_segments)} lines).")
        else:
            english_segments, _ = asr.transcribe(
                vocals_path, language=source_language.code, model_size=whisper_model,
                task="translate", log=log,
            )
            english_path.write_text(json.dumps(
                {"segments": [s.to_dict() for s in english_segments]}, ensure_ascii=False, indent=2,
            ), encoding="utf-8")
        if not english_segments:
            raise RuntimeError("The speech-to-English pass produced no text.")
        for segment in english_segments:
            # Attach the source-language words spoken in this time range (subtitles, cloning).
            overlap = [s.text for s in segments if min(s.end, segment.end) - max(s.start, segment.start) > 0.2]
            segment.translation = segment.text
            segment.text = " ".join(overlap) or segment.text

    english_texts = [segment.translation for segment in english_segments]

    # 4. Translate + speak each target language ------------------------------------
    from .tts import create_engine

    translator = None
    if any(target.code not in ("en", source_language.code) for target in target_languages):
        from .translate import Translator

        progress(35, "Loading translator")
        translator = Translator(nllb_model, log=log)
    reference_clips = (_reference_clips(vocals_path, english_segments, work_dir / "reference_clips")
                       if tts_engine == "voxcpm" else None)
    stem = _safe_name(video.stem)
    results = []
    span = 60 / len(target_languages)

    for number, target in enumerate(target_languages):
        base = 35 + number * span
        progress(base + span * 0.05, f"Translating to {target.name}")
        if target.code == source_language.code:
            dub_segments = segments
            texts = [segment.text for segment in dub_segments]
        elif target.code == "en":
            dub_segments, texts = english_segments, english_texts
        else:
            dub_segments = english_segments
            texts = translator.translate(english_texts, english, target)
        for segment, text in zip(dub_segments, texts):
            segment.translation = text
            log(f"  [{segment.start:6.2f}s] {text}")

        progress(base + span * 0.2, f"Generating {target.name} speech")
        engine = create_engine(target, tts_engine, gender, log=log)
        voice_track = dub.render_voice_track(
            dub_segments, engine, info["duration"], work_dir, reference_clips=reference_clips, log=log,
        )

        progress(base + span * 0.6, f"Mixing {target.name} audio")
        out_dir = job_dir / target.code
        out_dir.mkdir(exist_ok=True)
        name = f"{stem}_{target.code}"
        files = {
            "video": out_dir / f"{name}.mp4",
            "audio_wav": out_dir / f"{name}_audio.wav",
            "audio_mp3": out_dir / f"{name}_audio.mp3",
            "voice_only": out_dir / f"{name}_voice_only.wav",
            "subtitles": out_dir / f"{name}.srt",
        }
        voice_track = dub.mix(voice_track, vocals_path if background_path else None, background_path,
                              dub_segments, files["audio_wav"])
        sf.write(files["voice_only"], np.clip(voice_track, -1, 1), dub.SAMPLE_RATE, subtype="PCM_16")
        media.ffmpeg("-i", files["audio_wav"], "-c:a", "libmp3lame", "-b:a", "192k", files["audio_mp3"])
        write_srt(dub_segments, files["subtitles"])
        (out_dir / "transcript.json").write_text(json.dumps(
            {"source": source_language.code, "target": target.code, "segments": [s.to_dict() for s in dub_segments]},
            ensure_ascii=False, indent=2,
        ), encoding="utf-8")

        if info["has_video"]:
            progress(base + span * 0.7, f"Writing {target.name} video")
            mux_video(video, files["audio_wav"], files["video"])
            if lip_sync:
                from .lipsync import lip_sync as run_lip_sync

                progress(base + span * 0.75, f"Lip syncing {target.name} video")
                files["video_lipsync"] = out_dir / f"{name}_lipsync.mp4"
                run_lip_sync(video, files["audio_wav"], files["video_lipsync"], dub_segments,
                             voice_track, dub.SAMPLE_RATE, work_dir, original_audio=vocals_path, log=log)
        else:
            files.pop("video")

        results.append({"language": target.name, "code": target.code,
                        "files": {kind: str(path) for kind, path in files.items()}})
        del engine

    summary = {
        "video": str(video),
        "job_dir": str(job_dir),
        "source_language": source_language.code,
        "voice": gender,
        "lines": len(segments),
        "seconds": round(time.time() - started, 1),
        "outputs": results,
    }
    (job_dir / "result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    progress(100, f"Finished in {summary['seconds']}s")
    print("RESULT " + json.dumps(summary, ensure_ascii=False), flush=True)
    return summary
