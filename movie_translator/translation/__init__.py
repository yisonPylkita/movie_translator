"""Translation module for AI-powered subtitle translation."""

from .models import DEFAULT_BATCH_SIZE, DEFAULT_DEVICE, DEFAULT_MODEL, TRANSLATION_MODELS
from .translator import SubtitleTranslator, translate_dialogue_lines

__all__ = [
    'SubtitleTranslator',
    'translate_dialogue_lines',
    'TRANSLATION_MODELS',
    'DEFAULT_MODEL',
    'DEFAULT_DEVICE',
    'DEFAULT_BATCH_SIZE',
]
