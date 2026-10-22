"""Command-line entry point for the offline video dubbing pipeline.

Examples:
    python Video_Translator.py movie.mp4 --target en
    python Video_Translator.py movie.mp4 --source te --target en hi es --lip-sync
    python Video_Translator.py --download-models --target en hi
"""

import argparse
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Idle PyTorch/OpenMP threads must sleep, not spin, or they starve ONNX Runtime (GFPGAN).
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from dubbing.languages import LANGUAGES, SOURCE_CHOICES, TARGET_CHOICES  # noqa: E402

# Kept for the web GUI, which builds its language dropdowns from these.
SOURCE_LANGUAGES = SOURCE_CHOICES
TARGET_LANGUAGES = TARGET_CHOICES


def parse_args():
    parser = argparse.ArgumentParser(description="Offline video translation and dubbing (no API keys)")
    parser.add_argument("video", nargs="?", help="Path to the input video (or audio) file")
    parser.add_argument("--source", "-s", "--source-language", default="auto", help="Spoken language, e.g. te, or auto")
    parser.add_argument("--target", "-t", "--target-language", nargs="+", default=["en"],
                        help="One or more target languages, e.g. en hi es")
    parser.add_argument("--whisper-model", default="medium", choices=["small", "medium", "large-v3"],
                        help="Speech recognition size: small is fastest, large-v3 is most accurate")
    parser.add_argument("--generic-whisper", action="store_true",
                        help="Do not switch to the language-specific Indic Whisper models")
    parser.add_argument("--translator", default="600M", choices=["600M", "1.3B"], help="NLLB translation model size")
    parser.add_argument("--tts", default="auto", choices=["auto", "piper", "mms", "voxcpm"],
                        help="Speech engine. voxcpm clones the original voice but is very slow on CPU")
    parser.add_argument("--voice", default="auto", choices=["auto", "male", "female"],
                        help="Dubbing voice; auto picks from the original speaker's pitch")
    parser.add_argument("--no-separation", action="store_true", help="Skip music/voice separation (faster)")
    parser.add_argument("--lip-sync", action="store_true", help="Also render a Wav2Lip lip-synced video")
    parser.add_argument("--lip-sync-quality", default="best", choices=["best", "fast"],
                        help="best: GFPGAN HD mouth (demo quality, ~2 s/frame on CPU). fast: quick draft preview")
    parser.add_argument("--download-models", action="store_true", help="Download the models and exit")
    parser.add_argument("--list-languages", action="store_true", help="Print supported languages and exit")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list_languages:
        for language in LANGUAGES:
            engine = "piper" if language.piper else "mms" if language.mms else "source only"
            print(f"{language.code:3} {language.name:20} voice: {engine}")
        return
    if args.download_models:
        from dubbing.models import prefetch

        prefetch(whisper=args.whisper_model, nllb=args.translator, targets=args.target, lip_sync=args.lip_sync)
        return
    if not args.video:
        sys.exit("Please pass a video file. Run with --help for options.")

    from dubbing.pipeline import run

    run(
        args.video,
        targets=args.target,
        source=args.source,
        whisper_model=args.whisper_model,
        use_indic=not args.generic_whisper,
        nllb_model=args.translator,
        tts_engine=args.tts,
        voice=args.voice,
        separate=not args.no_separation,
        lip_sync=args.lip_sync,
        lip_sync_quality=args.lip_sync_quality,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Cancelled.")
    except Exception as exc:
        print(f"ERROR: {exc}", flush=True)
        raise
