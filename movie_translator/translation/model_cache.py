"""Centralized cache for translation model instances.

Replaces the module-level globals (_cached_translator, _cached_apple_backend)
with an injectable class that can be shared across pipeline runs and passed
to stages and GPU tasks explicitly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..logging import logger

if TYPE_CHECKING:
    from .apple_backend import AppleTranslationBackend
    from .translator import SubtitleTranslator


class ModelCache:
    """Owns cached translation model instances.

    Create one per pipeline run and pass it to translate_dialogue_lines(),
    TranslateStage, and TranslateTask.  The same instance should be reused
    across all files in a run so models are loaded only once.
    """

    def __init__(self) -> None:
        self._translator: SubtitleTranslator | None = None
        self._apple_backend: AppleTranslationBackend | None = None

    def get_translator(
        self, device: str, batch_size: int, model: str
    ) -> tuple[SubtitleTranslator | None, bool]:
        """Return a cached translator, reloading only when config changes.

        Returns (translator, cached) where cached is True if the model
        was already loaded with matching config.
        """
        from .translator import SubtitleTranslator

        if (
            self._translator is not None
            and self._translator.model is not None
            and self._translator.device == ('mps' if device == 'mps' else 'cpu')
            and self._translator.batch_size == batch_size
            and self._translator.model_key == model
        ):
            self._translator.preprocessing_stats.reset()
            return self._translator, True

        if self._translator is not None:
            self._translator.cleanup()

        translator = SubtitleTranslator(device=device, batch_size=batch_size, model_key=model)
        if not translator.load_model():
            return None, False
        self._translator = translator
        return translator, False

    def get_apple_backend(self, batch_size: int) -> AppleTranslationBackend | None:
        """Return a cached Apple backend instance."""
        from .apple_backend import AppleTranslationBackend

        if self._apple_backend is not None and self._apple_backend.batch_size == batch_size:
            self._apple_backend.preprocessing_stats.reset()
            return self._apple_backend

        try:
            backend = AppleTranslationBackend(batch_size=batch_size)
            self._apple_backend = backend
            return backend
        except (FileNotFoundError, RuntimeError) as e:
            logger.error(f'Apple Translation backend unavailable: {e}')
            return None

    def cleanup(self) -> None:
        """Release all cached models."""
        if self._translator is not None:
            self._translator.cleanup()
            self._translator = None
        if self._apple_backend is not None:
            self._apple_backend.cleanup()
            self._apple_backend = None
