"""Language table shared by speech recognition, translation and speech synthesis.

Every engine names languages differently, so each row maps one language to:
  - code:  ISO 639-1 code used by Whisper (and the CLI / web GUI)
  - nllb:  FLORES-200 code used by the NLLB translation model
  - piper: (male voice, female voice) from rhasspy/piper-voices, or None
  - mms:   ISO 639-3 code for facebook/mms-tts-* (fallback when Piper has no voice)
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Language:
    name: str
    code: str
    nllb: str
    piper: Optional[Tuple[str, str]] = None
    mms: Optional[str] = None

    @property
    def can_speak(self):
        return bool(self.piper or self.mms)


LANGUAGES = [
    # Indian languages
    Language("Telugu", "te", "tel_Telu", ("te_IN-venkatesh-medium", "te_IN-maya-medium"), "tel"),
    Language("Hindi", "hi", "hin_Deva", ("hi_IN-rohan-medium", "hi_IN-priyamvada-medium"), "hin"),
    Language("Tamil", "ta", "tam_Taml", None, "tam"),
    Language("Kannada", "kn", "kan_Knda", None, "kan"),
    Language("Malayalam", "ml", "mal_Mlym", ("ml_IN-arjun-medium", "ml_IN-meera-medium"), "mal"),
    Language("Marathi", "mr", "mar_Deva", ("mr_IN-google-medium", "mr_IN-google-medium"), "mar"),
    Language("Bengali", "bn", "ben_Beng", ("bn_BD-google-medium", "bn_BD-google-medium"), "ben"),
    Language("Gujarati", "gu", "guj_Gujr", None, "guj"),
    Language("Punjabi", "pa", "pan_Guru", None, "pan"),
    Language("Urdu", "ur", "urd_Arab", ("ur_PK-fasih-medium", "ur_PK-aegis_female-medium"), "urd-script_arabic"),
    Language("Nepali", "ne", "npi_Deva", ("ne_NP-chitwan-medium", "ne_NP-chitwan-medium"), None),
    # Global languages
    Language("English", "en", "eng_Latn", ("en_US-ryan-medium", "en_US-amy-medium"), "eng"),
    Language("Spanish", "es", "spa_Latn", ("es_ES-davefx-medium", "es_AR-daniela-high"), "spa"),
    Language("French", "fr", "fra_Latn", ("fr_FR-tom-medium", "fr_FR-siwis-medium"), "fra"),
    Language("German", "de", "deu_Latn", ("de_DE-thorsten-medium", "de_DE-kerstin-low"), "deu"),
    Language("Italian", "it", "ita_Latn", ("it_IT-riccardo-x_low", "it_IT-paola-medium"), None),
    Language("Portuguese", "pt", "por_Latn", ("pt_BR-faber-medium", "pt_BR-faber-medium"), "por"),
    Language("Russian", "ru", "rus_Cyrl", ("ru_RU-dmitri-medium", "ru_RU-irina-medium"), "rus"),
    Language("Arabic", "ar", "arb_Arab", ("ar_JO-kareem-medium", "ar_JO-kareem-medium"), "ara"),
    Language("Turkish", "tr", "tur_Latn", ("tr_TR-dfki-medium", "tr_TR-dfki-medium"), "tur"),
    Language("Dutch", "nl", "nld_Latn", ("nl_NL-pim-medium", "nl_BE-nathalie-medium"), "nld"),
    Language("Polish", "pl", "pol_Latn", ("pl_PL-darkman-medium", "pl_PL-gosia-medium"), "pol"),
    Language("Chinese (Mandarin)", "zh", "zho_Hans", ("zh_CN-chaowen-medium", "zh_CN-huayan-medium"), None),
    Language("Japanese", "ja", "jpn_Jpan", None, None),
    Language("Korean", "ko", "kor_Hang", ("ko_KR-kss-medium", "ko_KR-kss-medium"), "kor"),
    Language("Indonesian", "id", "ind_Latn", ("id_ID-news_tts-medium", "id_ID-news_tts-medium"), "ind"),
    Language("Vietnamese", "vi", "vie_Latn", ("vi_VN-vais1000-medium", "vi_VN-vais1000-medium"), "vie"),
    Language("Thai", "th", "tha_Thai", ("th_TH-tsync2-medium", "th_TH-tsync2-medium"), "tha"),
    Language("Ukrainian", "uk", "ukr_Cyrl", ("uk_UA-ukrainian_tts-medium", "uk_UA-ukrainian_tts-medium"), "ukr"),
    Language("Persian", "fa", "pes_Arab", ("fa_IR-amir-medium", "fa_IR-gyro-medium"), "fas"),
    Language("Swahili", "sw", "swh_Latn", ("sw_CD-lanfrica-medium", "sw_CD-lanfrica-medium"), "swh"),
]

BY_CODE = {language.code: language for language in LANGUAGES}

SOURCE_CHOICES = {"Auto Detect": "auto", **{language.name: language.code for language in LANGUAGES}}
TARGET_CHOICES = {language.name: language.code for language in LANGUAGES if language.can_speak}


def get_language(value, allow_auto=False):
    """Resolve a language code or name ("te", "Telugu") to a Language, or None for auto."""
    if value is None:
        if allow_auto:
            return None
        raise ValueError("A target language is required.")
    text = str(value).strip().lower()
    if allow_auto and text in {"", "auto", "automatic", "auto detect"}:
        return None
    for language in LANGUAGES:
        if text in {language.code, language.name.lower()}:
            return language
    valid = ", ".join(f"{language.name}={language.code}" for language in LANGUAGES)
    raise ValueError(f"Unsupported language: {value!r}. Supported: {valid}")
