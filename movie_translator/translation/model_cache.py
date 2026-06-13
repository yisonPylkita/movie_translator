"""Centralized cache for translation model instances.

Replaces the module-level globals (_cached_translator, _cached_apple_backend)
with an injectable class that can be shared across pipeline runs and passed
to stages and GPU tasks explicitly.

Supports two backends:
- ``'mlx'``: Apple Silicon MLX (Metal-native, Allegro BiDi model)
- ``'apple'``: macOS Translation framework via Rust + Swift bridge (no Python)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..logging import logger

if TYPE_CHECKING:
    from .mlx_backend import BidiMLXModel


class _MlxTranslatorWrapper:
    """Thin wrapper adapting BidiMLXModel to SubtitleTranslator-like interface.

    Exposes translate_texts() so the pipeline can use it.
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
    """Owns cached MLX model instance.

    Create one per pipeline run and pass it to translate_dialogue_lines(),
    TranslateStage, and TranslateTask.  The same instance should be reused
    across all files in a run so models are loaded only once.

    Only the MLX backend is cached in Python.  The Apple Translation backend
    is handled entirely in Rust (calling the Swift bridge directly).
    """

    def __init__(self) -> None:
        self._mlx_model: BidiMLXModel | None = None

    def get_translator(
        self, device: str, batch_size: int, model: str
    ) -> tuple[_MlxTranslatorWrapper | None, bool]:
        """Return a cached MLX translator, reloading only when config changes.

        Returns (translator, cached) where cached is True if the model
        was already loaded with matching config.
        """
        from .mlx_backend import BidiMLXModel as BidiMLXModelCls

        if model != 'mlx':
            logger.error(f'Unknown model backend: {model!r}')
            return None, False

        if self._mlx_model is not None:
            return _MlxTranslatorWrapper(self._mlx_model, batch_size), True

        model_instance = BidiMLXModelCls()
        try:
            model_instance.load_mlx_weights()
        except Exception as e:
            logger.error(f'Failed to load MLX model: {e}')
            return None, False

        self._mlx_model = model_instance
        return _MlxTranslatorWrapper(model_instance, batch_size), False

    def cleanup(self) -> None:
        """Release all cached models."""
        self._mlx_model = None
