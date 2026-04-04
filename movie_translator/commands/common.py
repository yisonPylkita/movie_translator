"""Shared utilities for CLI commands."""

import sys

from ..logging import console, logger


def check_dependencies() -> bool:
    """Check all required dependencies. Returns True if all satisfied."""
    import importlib.util

    from ..ffmpeg import get_ffmpeg_version

    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        console.print(f'[red]❌ Python 3.10+ required, found {version.major}.{version.minor}[/red]')
        return False

    try:
        get_ffmpeg_version()
    except Exception:
        console.print('[red]❌ FFmpeg not available. Run ./setup.sh first.[/red]')
        return False

    required_packages = ['pysubs2', 'torch', 'transformers']
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            console.print(f'[red]❌ Missing package: {package}. Run ./setup.sh first.[/red]')
            return False

    return True


def resolve_model(explicit_choice: str | None) -> str:
    """Pick translation backend: explicit choice, or auto-detect Apple, or Allegro."""
    if explicit_choice is not None:
        return explicit_choice

    try:
        from movie_translator.translation.apple_backend import (
            check_languages_installed,
            is_available,
        )

        if is_available() and check_languages_installed():
            logger.info('Apple Translation available — using on-device backend')
            return 'apple'
    except Exception:
        pass

    return 'allegro'
