from movie_translator.translator_adapter import translate_file
from movie_translator.main import MovieProcessor
from movie_translator.config import TranslationConfig, setup_logging
from movie_translator.exceptions import (
    MovieTranslatorError,
    SubtitleNotFoundError,
    TranslationError,
    MKVProcessingError,
    ConfigurationError,
)

__all__ = [
    "translate_file",
    "MovieProcessor",
    "TranslationConfig",
    "setup_logging",
    "MovieTranslatorError",
    "SubtitleNotFoundError",
    "TranslationError",
    "MKVProcessingError",
    "ConfigurationError",
]

__version__ = "0.1.0"
