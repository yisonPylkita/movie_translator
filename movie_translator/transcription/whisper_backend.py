"""mlx-whisper large-v3 backend (Apple MLX / Metal).

The cross-engine fallback from the bake-off: accurate on EN and JA, Metal-fast.
Raw Whisper hallucinates on trailing music/silence, so callers MUST run
`postfilter.clean_segments` on the output (the pipeline entry point does).
Model weights pull from HF (mlx-community/whisper-large-v3-mlx) on first use.
"""

from __future__ import annotations

from pathlib import Path

from ..logging import logger
from ..types import DialogueLine

MODEL_REPO = 'mlx-community/whisper-large-v3-mlx'


def is_available() -> bool:
    try:
        import mlx_whisper  # noqa: F401

        return True
    except ImportError:
        return False


def transcribe(wav: Path, language: str) -> list[DialogueLine]:
    """Transcribe `wav` -> raw Whisper segments (un-filtered)."""
    import mlx_whisper

    logger.info(f'mlx-whisper ({MODEL_REPO}) transcribing {wav.name} ({language})')
    result = mlx_whisper.transcribe(
        str(wav),
        path_or_hf_repo=MODEL_REPO,
        language=language,
        # Reduces repetition-loop hallucinations on music/silence.
        condition_on_previous_text=False,
        verbose=None,
    )
    return [
        DialogueLine(int(s['start'] * 1000), int(s['end'] * 1000), s['text'].strip())
        for s in result['segments']
        if s['text'].strip()
    ]
