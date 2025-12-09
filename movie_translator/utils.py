#!/usr/bin/env python3
"""
Utility functions for the Movie Translator pipeline.
"""

import gc

from rich.console import Console

console = Console()


def clear_memory():
    """Clear memory caches and force garbage collection."""
    gc.collect()


def replace_polish_chars(text: str) -> str:
    """Replace Polish characters with English equivalents."""
    polish_to_english = {
        'ą': 'a',
        'ć': 'c',
        'ę': 'e',
        'ł': 'l',
        'ń': 'n',
        'ó': 'o',
        'ś': 's',
        'ź': 'z',
        'ż': 'z',
        'Ą': 'A',
        'Ć': 'C',
        'Ę': 'E',
        'Ł': 'L',
        'Ń': 'N',
        'Ó': 'O',
        'Ś': 'S',
        'Ź': 'Z',
        'Ż': 'Z',
    }

    for polish, english in polish_to_english.items():
        text = text.replace(polish, english)

    return text


def log_info(message: str):
    """Log info message."""
    console.print(f'[INFO] {message}')


def log_success(message: str):
    """Log success message."""
    console.print(f'[SUCCESS] {message}')


def log_warning(message: str):
    """Log warning message."""
    console.print(f'[WARNING] {message}')


def log_error(message: str):
    """Log error message."""
    console.print(f'[ERROR] {message}')
