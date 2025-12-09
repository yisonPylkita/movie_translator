__version__ = '1.0.0'
__author__ = 'Movie Translator Team'

from .logging import logger
from .main import main
from .pipeline import TranslationPipeline
from .types import DialogueLine, SubtitleFile

__all__ = [
    'DialogueLine',
    'SubtitleFile',
    'TranslationPipeline',
    'logger',
    'main',
]
