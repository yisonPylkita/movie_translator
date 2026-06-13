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
    from .mlx_backend import BidiMLXModel
    from .translator import SubtitleTranslator


class _MlxTranslatorWrapper:
    """Thin wrapper adapting BidiMLXModel to SubtitleTranslator-like interface.

    Exposes translate_texts() so the pipeline can use it via the same
    translate_dialogue_lines() dispatch without special-casing.
    """

    def __init__(self, model: BidiMLXModel, batch_size: int) -> None:
        self.model = model
        self.batch_size = batch_size
        self.proper_nouns: set[str] = set()

    def translate_texts(self, texts: list[str], progress_callback=None) -> list[str]:
        return self.model.translate(
            texts,
            max_new_tokens=128,
            progress_callback=progress_callback,
            batch_size=self.batch_size,
        )

    def cleanup(self) -> None:
        pass


class ModelCache:
    """Owns cached translation model instances.

    Create one per pipeline run and pass it to translate_dialogue_lines(),
    TranslateStage, and TranslateTask.  The same instance should be reused
    across all files in a run so models are loaded only once.

    Supports three backends:
    - ``'allegro'`` (default): PyTorch-based SubtitleTranslator
    - ``'apple'``: macOS Translation framework via Swift bridge
    - ``'mlx'``: Apple Silicon MLX (Metal-native, best perf)
    """

    def __init__(self) -> None:
        self._translator: SubtitleTranslator | None = None
        self._apple_backend: AppleTranslationBackend | None = None
        self._mlx_model: BidiMLXModel | None = None

    def get_translator(
        self, device: str, batch_size: int, model: str
    ) -> tuple[SubtitleTranslator | _MlxTranslatorWrapper | None, bool]:
        """Return a cached translator, reloading only when config changes.

        Returns (translator, cached) where cached is True if the model
        was already loaded with matching config.

        When *model* is ``'mlx'``, returns a _MlxTranslatorWrapper instead
        of a SubtitleTranslator.
        """
        from .translator import SubtitleTranslator

        # MLX backend is handled separately
        if model == 'mlx':
            return self._get_mlx_wrapper(batch_size)

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

    def _get_mlx_wrapper(self, batch_size: int) -> tuple[_MlxTranslatorWrapper | None, bool]:
        """Return a cached MLX model wrapper."""
        from .mlx_backend import BidiMLXModel as BidiMLXModelCls

        if self._mlx_model is not None:
            return _MlxTranslatorWrapper(self._mlx_model, batch_size), True

        model = BidiMLXModelCls()
        try:
            model.load_mlx_weights()
        except Exception as e:
            logger.error(f'Failed to load MLX model: {e}')
            return None, False

        self._mlx_model = model
        return _MlxTranslatorWrapper(model, batch_size), False

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
        self._mlx_model = None
