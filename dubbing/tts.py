"""Text-to-speech engines.

piper  - default. Fast neural TTS (ONNX) with natural voices for ~30 of our languages.
mms    - Meta MMS-TTS fallback for languages Piper lacks (Tamil, Kannada, Gujarati, Punjabi).
voxcpm - optional voice cloning from the original speaker. Very slow without an NVIDIA GPU.

Every engine exposes synthesize(text, speed=1.0, **kwargs) -> (float32 mono audio, sample_rate).
"""

import numpy as np

from . import models


class PiperEngine:
    name = "piper"

    def __init__(self, voice_name, log=print):
        from piper import PiperVoice

        log(f"Loading Piper voice {voice_name}...")
        self.voice = PiperVoice.load(str(models.piper_voice(voice_name)))
        self.sample_rate = self.voice.config.sample_rate
        self.base_length_scale = getattr(self.voice.config, "length_scale", 1.0) or 1.0

    def synthesize(self, text, speed=1.0, **_):
        from piper import SynthesisConfig

        config = SynthesisConfig(
            speaker_id=0 if self.voice.config.num_speakers > 1 else None,
            length_scale=self.base_length_scale / speed,
            noise_scale=0.6,
            noise_w_scale=0.8,
        )
        chunks = [chunk.audio_float_array for chunk in self.voice.synthesize(text, config)]
        gap = np.zeros(int(self.sample_rate * 0.12), dtype=np.float32)
        audio = np.concatenate([part for chunk in chunks for part in (chunk, gap)][:-1]) if chunks else gap
        return audio.astype(np.float32), self.sample_rate


class MMSEngine:
    name = "mms"

    def __init__(self, mms_code, log=print):
        import torch
        from transformers import AutoTokenizer, VitsModel

        log(f"Loading MMS-TTS voice {mms_code}...")
        path = models.mms_model(mms_code)
        self.torch = torch
        self.model = VitsModel.from_pretrained(str(path)).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.sample_rate = self.model.config.sampling_rate
        self.uroman = None
        if getattr(self.tokenizer, "is_uroman", False):
            import uroman

            self.uroman = uroman.Uroman()

    def synthesize(self, text, speed=1.0, **_):
        if self.uroman is not None:
            text = self.uroman.romanize_string(text)
        inputs = self.tokenizer(text, return_tensors="pt")
        self.model.speaking_rate = speed
        self.torch.manual_seed(0)
        with self.torch.inference_mode():
            audio = self.model(**inputs).waveform[0].numpy()
        return audio.astype(np.float32), self.sample_rate


class VoxCPMEngine:
    """Clones the original speaker's voice. Needs a reference clip per segment."""

    name = "voxcpm"

    def __init__(self, log=print):
        import sys

        sys.path.insert(0, str(models.BASE_DIR))
        from voxcpm_runtime import load_model, model_ready, VOXCPM_OUTPUT_SAMPLE_RATE

        if not model_ready()[0]:
            raise RuntimeError("VoxCPM2 model is not downloaded. Run: python scripts/setup_models.py")
        log("Loading VoxCPM2 voice-cloning model (slow on CPU)...")
        self.model = load_model()
        self.sample_rate = VOXCPM_OUTPUT_SAMPLE_RATE

    def synthesize(self, text, speed=1.0, reference_wav=None, prompt_text=None, seed=42, **_):
        import torch

        if reference_wav is None:
            raise ValueError("VoxCPM needs a reference_wav of the original speaker.")
        options = {"text": text, "reference_wav_path": str(reference_wav), "cfg_value": 2.0,
                   "inference_timesteps": 10, "normalize": False, "denoise": False}
        if prompt_text:
            options.update(prompt_wav_path=str(reference_wav), prompt_text=prompt_text)
        torch.manual_seed(seed)
        audio = np.asarray(self.model.generate(**options), dtype=np.float32)
        return audio, self.sample_rate


def create_engine(language, engine="auto", gender="male", log=print):
    if engine == "voxcpm":
        return VoxCPMEngine(log=log)
    if engine in ("auto", "piper") and language.piper:
        return PiperEngine(language.piper[1 if gender == "female" else 0], log=log)
    if engine in ("auto", "mms") and language.mms:
        return MMSEngine(language.mms, log=log)
    raise RuntimeError(f"No {engine} text-to-speech voice is available for {language.name}.")
