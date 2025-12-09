"""
Movie Translator - A modular pipeline for translating movie subtitles.

This package provides a complete solution for:
1. Extracting English dialogue from MKV files
2. AI translation to Polish
3. Creating clean MKV files with Polish as default subtitle track

Usage:
    from movie_translator import translate_mkv
    translate_mkv("/path/to/mkv", "/path/to/output")
"""

__version__ = '1.0.0'
__author__ = 'Movie Translator Team'

# Import main functions for easy access
from .main import main, process_mkv_file
from .utils import log_error, log_info, log_success, log_warning

__all__ = ['process_mkv_file', 'main', 'log_info', 'log_success', 'log_warning', 'log_error']
