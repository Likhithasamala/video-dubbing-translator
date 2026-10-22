"""Speech recognition with faster-whisper (CTranslate2, int8 on CPU)."""

import os
from dataclasses import dataclass, field, asdict

from . import models

SENTENCE_END = (".", "!", "?", "।", "॥", "。", "！", "？", "؟")
MAX_SEGMENT_SECONDS = 12.0
SPLIT_PAUSE_SECONDS = 0.7


@dataclass
class Segment:
    start: float
    end: float
    text: str
    translation: str = ""
    words: list = field(default_factory=list)

    @property
    def duration(self):
        return self.end - self.start

    def to_dict(self):
        data = asdict(self)
        data.pop("words")
        return data


def _device():
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", "float16"
    except ImportError:
        pass
    return "cpu", "int8"


def _load(path, log, label):
    from faster_whisper import WhisperModel

    device, compute_type = _device()
    log(f"Loading {label} on {device}...")
    # Physical cores only: hyper-threads slow CTranslate2 down on Ryzen/Intel laptops.
    threads = max(1, (os.cpu_count() or 8) // 2)
    return WhisperModel(str(path), device=device, compute_type=compute_type, cpu_threads=threads)


def transcribe(audio_path, language=None, model_size="medium", use_indic=True, task="transcribe", log=print):
    """Return (segments, detected_language_code).

    task="translate" makes Whisper listen and write English directly. That handles
    code-mixed speech (Telugu/Tamil sprinkled with English) far better than
    transcribing first and translating the text afterwards.

    For transcription of Telugu, Hindi, Tamil, Kannada and Gujarati a Whisper
    fine-tuned on that language is used, because generic Whisper garbles them.
    """
    options = dict(beam_size=5, vad_filter=True, vad_parameters={"min_silence_duration_ms": 400},
                   word_timestamps=True, condition_on_previous_text=False)

    if task == "translate" or language is None or not (use_indic and language in models.INDIC_WHISPER):
        model = _load(models.whisper_model(model_size), log, f"Whisper {model_size}")
        raw_segments, info = model.transcribe(str(audio_path), language=language, task=task, **options)
        if language is None:
            log(f"Detected language: {info.language} (confidence {info.language_probability:.2f})")
        language = info.language

    if task == "transcribe" and use_indic and language in models.INDIC_WHISPER:
        size = "small" if model_size == "small" else "medium"
        model = _load(models.indic_whisper_model(language, size), log,
                      f"{models.INDIC_WHISPER[language].title()} Whisper {size}")
        return _transcribe_by_phrase(model, audio_path, language, log), language

    words = []
    for segment in raw_segments:
        log(f"  [{segment.start:6.2f} - {segment.end:6.2f}] {segment.text.strip()}")
        if segment.words:
            words.extend((w.start, w.end, w.word) for w in segment.words if w.word.strip())
        elif segment.text.strip():
            words.append((segment.start, segment.end, segment.text))

    return _group_words(words), language


def _transcribe_by_phrase(model, audio_path, language, log):
    """Cut the audio at pauses and transcribe each phrase on its own.

    The fine-tuned Indic checkpoints were trained without timestamp tokens, so
    letting them decode a 30 s window returns one huge line that gets truncated.
    Short VAD phrases give complete text and exact start/end times instead.
    """
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    rate = 16000
    audio = decode_audio(str(audio_path), sampling_rate=rate)
    regions = get_speech_timestamps(
        audio, VadOptions(min_silence_duration_ms=300, speech_pad_ms=150, max_speech_duration_s=MAX_SEGMENT_SECONDS)
    )
    segments = []
    for region in regions:
        start, end = region["start"] / rate, region["end"] / rate
        if end - start < 0.3:
            continue
        pieces, _ = model.transcribe(audio[region["start"]:region["end"]], language=language, beam_size=5,
                                     without_timestamps=True, condition_on_previous_text=False)
        text = " ".join(piece.text.strip() for piece in pieces).strip()
        if text:
            log(f"  [{start:6.2f} - {end:6.2f}] {text}")
            segments.append(Segment(start, end, text))
    return segments


def _group_words(words):
    """Group words into sentence-like segments that are short enough to dub."""
    segments, current = [], []

    def flush():
        if current:
            text = "".join(w[2] for w in current).strip()
            if text:
                segments.append(Segment(current[0][0], current[-1][1], text, words=list(current)))
            current.clear()

    for word in words:
        if current:
            pause = word[0] - current[-1][1]
            too_long = word[1] - current[0][0] > MAX_SEGMENT_SECONDS
            if pause > SPLIT_PAUSE_SECONDS or too_long:
                flush()
        current.append(word)
        if word[2].strip().endswith(SENTENCE_END):
            flush()
    flush()

    # Fold tiny fragments ("Yes.", "Okay") into their neighbour so the TTS has context.
    merged = []
    for segment in segments:
        previous = merged[-1] if merged else None
        if (
            previous is not None
            and (segment.duration < 1.0 or previous.duration < 1.0)
            and segment.start - previous.end < 0.4
            and segment.end - previous.start <= MAX_SEGMENT_SECONDS
        ):
            previous.end = segment.end
            previous.text = f"{previous.text} {segment.text}"
            previous.words.extend(segment.words)
        else:
            merged.append(segment)
    return merged
