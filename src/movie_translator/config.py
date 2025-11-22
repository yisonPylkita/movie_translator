import logging
import sys
from dataclasses import dataclass
from typing import Optional

from movie_translator.constants import (
    DEFAULT_MODEL,
    DEFAULT_BATCH_SIZE,
    DEFAULT_SCENE_THRESHOLD,
    LANGUAGE_POLISH,
    LOG_FORMAT,
    DEFAULT_DEVICE,
)


@dataclass
class TranslationConfig:
    target_language: str = LANGUAGE_POLISH
    model: str = DEFAULT_MODEL
    batch_size: int = DEFAULT_BATCH_SIZE
    scene_threshold: float = DEFAULT_SCENE_THRESHOLD
    movie_name: Optional[str] = None
    device: str = DEFAULT_DEVICE

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.scene_threshold < 0:
            raise ValueError("Scene threshold must be non-negative")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level, format=LOG_FORMAT, handlers=[logging.StreamHandler(sys.stdout)]
    )
