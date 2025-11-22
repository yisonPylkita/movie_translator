# Model and device settings
DEFAULT_MODEL = "sdadas/flan-t5-base-translator-en-pl"  # Confirmed working EN->PL model
DEFAULT_DEVICE = "auto"  # Auto-detects MPS on M1/M2
DEFAULT_BATCH_SIZE = 16  # Optimized for M1 MacBook Air

LANGUAGE_POLISH = "Polish"
LANGUAGE_ENGLISH = "eng"
LANGUAGE_ENGLISH_SHORT = "en"

EXTENSION_MKV = ".mkv"
EXTENSION_SRT = ".srt"

TRACK_TYPE_SUBTITLE = "subtitles"

POLISH_TRACK_NAME = "Polish (AI)"

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
