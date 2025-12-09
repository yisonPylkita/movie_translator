__version__ = '1.0.0'
__author__ = 'Movie Translator Team'

from .logging import logger
from .main import main
from .pipeline import TranslationPipeline

__all__ = [
    'TranslationPipeline',
    'main',
    'logger',
]
