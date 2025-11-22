import logging
import sys
from dataclasses import dataclass

from movie_translator.constants import (
    DEFAULT_MODEL,
    DEFAULT_BATCH_SIZE,
    LANGUAGE_POLISH,
    LOG_FORMAT,
    DEFAULT_DEVICE,
)


@dataclass
class TranslationConfig:
    """Configuration for subtitle translation."""
    target_language: str = LANGUAGE_POLISH
    model: str = DEFAULT_MODEL
    batch_size: int = DEFAULT_BATCH_SIZE
    device: str = DEFAULT_DEVICE

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level, format=LOG_FORMAT, handlers=[logging.StreamHandler(sys.stdout)]
    )
