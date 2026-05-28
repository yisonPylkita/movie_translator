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
    """Back-compat: returns just the primary model. New code should use resolve_models()."""
    primary, _ = resolve_models(explicit_choice)
    return primary


def resolve_models(explicit_choice: str | None) -> tuple[str, list[str]]:
    """Pick the primary translation backend + any extra backends to also run.

    Returns (primary_model, extra_models). Each extra model produces an
    additional Polish subtitle track in the output MKV. On macOS where Apple
    Translation is available we default to running Allegro AND Apple so the
    user gets both tracks (Allegro keeps proper nouns intact, Apple has the
    higher chrF on plain dialogue).

    If the caller passed `--model X` explicitly we honor it and run nothing
    extra.
    """
    if explicit_choice is not None:
        return explicit_choice, []

    try:
        from movie_translator.translation.apple_backend import (
            check_languages_installed,
            is_available,
        )

        if is_available() and check_languages_installed():
            logger.info(
                'Apple Translation available — running Allegro + Apple (two PL tracks)'
            )
            return 'allegro', ['apple']
    except Exception:
        pass

    return 'allegro', []
