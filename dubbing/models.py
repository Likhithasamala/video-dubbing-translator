"""Download and locate the free, open models used by the pipeline.

Everything is fetched once from public Hugging Face repositories (no account,
no API key) and then runs fully offline from the local cache.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"

WHISPER_REPOS = {
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
}
# Languages with a dedicated fine-tuned Whisper (vasista22/whisper-<name>-<size>).
INDIC_WHISPER = {"te": "telugu", "hi": "hindi", "ta": "tamil", "kn": "kannada", "gu": "gujarati"}
NLLB_REPOS = {
    "600M": "JustFrederik/nllb-200-distilled-600M-ct2-int8",
    "1.3B": "JustFrederik/nllb-200-distilled-1.3B-ct2-int8",
}
PIPER_REPO = "rhasspy/piper-voices"
MMS_REPO_PREFIX = "facebook/mms-tts-"
WAV2LIP_REPO = "camenduru/Wav2Lip"
GFPGAN_REPO = "facefusion/models-3.0.0"
WAV2LIP_CHECKPOINT = "checkpoints/wav2lip_gan.pth"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)


def _local_dir(repo_id):
    # Plain folders under ./models: no symlinks (Windows needs admin rights for those)
    # and the whole project folder can be copied to another machine as-is.
    return MODELS_DIR / repo_id.replace("/", "--")


def _snapshot(repo_id, allow_patterns=None):
    from huggingface_hub import snapshot_download

    target = _local_dir(repo_id)
    marker = target / ".complete"
    if not marker.exists():
        print(f"Downloading {repo_id} (first run only)...", flush=True)
        snapshot_download(repo_id, allow_patterns=allow_patterns, local_dir=target)
        marker.touch()
    return target


def _file(repo_id, filename):
    from huggingface_hub import hf_hub_download

    path = _local_dir(repo_id) / filename
    if not path.exists():
        print(f"Downloading {repo_id}/{filename} (first run only)...", flush=True)
        hf_hub_download(repo_id, filename, local_dir=_local_dir(repo_id))
    return path


def whisper_model(size):
    if size not in WHISPER_REPOS:
        raise ValueError(f"Unknown Whisper model {size!r}. Choose one of: {', '.join(WHISPER_REPOS)}")
    return _snapshot(WHISPER_REPOS[size])


def indic_whisper_model(language_code, size="medium"):
    """Whisper fine-tuned for one Indian language (IIT Madras SPRING Lab, Apache-2.0).

    Generic Whisper is weak on Telugu, Tamil, etc. These checkpoints are much better,
    but only ship PyTorch weights, so they are converted to CTranslate2 int8 once.
    """
    name = INDIC_WHISPER[language_code]
    target = MODELS_DIR / f"whisper-{name}-{size}-ct2"
    if (target / "model.bin").exists() and (target / "tokenizer.json").exists():
        return target

    import shutil
    from ctranslate2.converters import TransformersConverter
    from transformers import WhisperTokenizerFast

    repo = f"vasista22/whisper-{name}-{size}"
    source = _snapshot(repo, allow_patterns=["*.json", "*.txt", "pytorch_model.bin"])
    print(f"Converting {repo} for fast CPU inference (one time)...", flush=True)
    WhisperTokenizerFast.from_pretrained(str(source)).save_pretrained(str(source))
    TransformersConverter(str(source), copy_files=["tokenizer.json", "preprocessor_config.json"]).convert(
        str(target), quantization="int8", force=True
    )
    shutil.rmtree(source, ignore_errors=True)  # the 3 GB float32 original is no longer needed
    return target


def nllb_model(size):
    if size not in NLLB_REPOS:
        raise ValueError(f"Unknown NLLB model {size!r}. Choose one of: {', '.join(NLLB_REPOS)}")
    return _snapshot(NLLB_REPOS[size])


def piper_voice(voice_name):
    """Return the .onnx path of a Piper voice such as 'te_IN-venkatesh-medium'."""
    locale, speaker, quality = voice_name.split("-", 2)
    folder = f"{locale.split('_')[0]}/{locale}/{speaker}/{quality}"
    onnx_path = _file(PIPER_REPO, f"{folder}/{voice_name}.onnx")
    _file(PIPER_REPO, f"{folder}/{voice_name}.onnx.json")
    return onnx_path


def mms_model(code):
    return _snapshot(
        MMS_REPO_PREFIX + code,
        allow_patterns=["config.json", "model.safetensors", "tokenizer_config.json", "vocab.json", "special_tokens_map.json"],
    )


def gfpgan_model():
    """GFPGAN v1.4 face restoration (Apache-2.0), ONNX export from the FaceFusion project."""
    return _file(GFPGAN_REPO, "gfpgan_1.4.onnx")


def syncnet_checkpoint():
    return _file(WAV2LIP_REPO, "checkpoints/lipsync_expert.pth")


def wav2lip_checkpoint():
    return _file(WAV2LIP_REPO, WAV2LIP_CHECKPOINT)


def yunet_model():
    path = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
    if not path.exists() or path.stat().st_size < 100_000:
        import requests

        print("Downloading YuNet face detector (first run only)...", flush=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        response = requests.get(YUNET_URL, timeout=120)
        response.raise_for_status()
        path.write_bytes(response.content)
    return path


def prefetch(whisper="medium", nllb="600M", targets=("en",), lip_sync=True):
    """Download everything a typical run needs so the first job does not stall."""
    from .languages import get_language

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    whisper_model(whisper)
    nllb_model(nllb)
    for code in targets:
        language = get_language(code)
        if language.piper:
            for voice in sorted(set(language.piper)):
                piper_voice(voice)
        elif language.mms:
            mms_model(language.mms)
    if lip_sync:
        wav2lip_checkpoint()
        syncnet_checkpoint()
        yunet_model()
        gfpgan_model()
    print("All models are ready.", flush=True)
