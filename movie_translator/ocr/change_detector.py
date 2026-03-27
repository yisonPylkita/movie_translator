from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from ..logging import logger


@dataclass
class SubtitleTransition:
    timestamp_ms: int
    frame_path: Path
    event_type: Literal['appeared', 'disappeared']


def _load_grayscale(image_path: Path) -> np.ndarray:
    """Load JPEG as grayscale numpy array without OpenCV."""
    from PIL import Image

    img = Image.open(image_path).convert('L')
    return np.array(img)


def _frame_has_text(frame: np.ndarray, variance_threshold: float = 200.0) -> bool:
    """Heuristic: frames with text have higher pixel variance than blank frames."""
    return float(np.var(frame)) > variance_threshold


def detect_transitions(
    frames: list[tuple[Path, int]],
    change_threshold: float = 15.0,
) -> list[SubtitleTransition]:
    if len(frames) < 2:
        return []

    transitions: list[SubtitleTransition] = []
    prev_frame = _load_grayscale(frames[0][0])
    prev_has_text = _frame_has_text(prev_frame)

    for i in range(1, len(frames)):
        frame_path, timestamp_ms = frames[i]
        curr_frame = _load_grayscale(frame_path)

        # Compute mean absolute pixel difference
        diff = np.mean(np.abs(curr_frame.astype(np.int16) - prev_frame.astype(np.int16)))

        if diff > change_threshold:
            curr_has_text = _frame_has_text(curr_frame)

            if curr_has_text and not prev_has_text:
                transitions.append(SubtitleTransition(timestamp_ms, frame_path, 'appeared'))
            elif not curr_has_text and prev_has_text:
                transitions.append(SubtitleTransition(timestamp_ms, frame_path, 'disappeared'))
            elif curr_has_text and prev_has_text:
                # Text changed — treat as new subtitle appeared
                transitions.append(SubtitleTransition(timestamp_ms, frame_path, 'appeared'))

            prev_has_text = curr_has_text

        prev_frame = curr_frame

    logger.info(f'Detected {len(transitions)} subtitle transitions')
    return transitions
