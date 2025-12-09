"""
Movie Translator - A modular pipeline for translating movie subtitles.

This package provides a complete solution for:
1. Extracting English dialogue from MKV files
2. AI translation to Polish
3. Creating clean MKV files with Polish as default subtitle track

Usage:
    from movie_translator import TranslationPipeline

    pipeline = TranslationPipeline(device='mps', batch_size=16)
    pipeline.process_mkv_file(mkv_path, output_dir)
"""

__version__ = '1.0.0'
__author__ = 'Movie Translator Team'

from .main import main
from .pipeline import TranslationPipeline
from .utils import log_error, log_info, log_success, log_warning

__all__ = [
    'TranslationPipeline',
    'main',
    'log_info',
    'log_success',
    'log_warning',
    'log_error',
]
