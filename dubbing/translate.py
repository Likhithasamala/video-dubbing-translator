"""Offline translation with Meta's NLLB-200 (CTranslate2 int8, runs on CPU)."""

import os

from . import models


class Translator:
    def __init__(self, size="600M", log=print):
        import ctranslate2
        from transformers import AutoTokenizer

        path = models.nllb_model(size)
        log(f"Loading NLLB-200 {size} translator...")
        self._translator = ctranslate2.Translator(
            str(path), device="cpu", compute_type="int8", intra_threads=os.cpu_count() or 4
        )
        self._tokenizer = AutoTokenizer.from_pretrained(str(path))

    def translate(self, texts, source, target):
        """Translate a list of strings between two Language objects."""
        if not texts:
            return []
        if source.code == target.code:
            return list(texts)
        self._tokenizer.src_lang = source.nllb
        batch = [self._tokenizer.convert_ids_to_tokens(self._tokenizer.encode(text)) for text in texts]
        results = self._translator.translate_batch(
            batch,
            target_prefix=[[target.nllb]] * len(batch),
            beam_size=4,
            max_batch_size=16,
            max_decoding_length=256,
            repetition_penalty=1.1,
            no_repeat_ngram_size=4,
        )
        outputs = []
        for result in results:
            tokens = result.hypotheses[0]
            if tokens and tokens[0] == target.nllb:
                tokens = tokens[1:]
            ids = self._tokenizer.convert_tokens_to_ids(tokens)
            outputs.append(self._tokenizer.decode(ids, skip_special_tokens=True).strip())
        return outputs
