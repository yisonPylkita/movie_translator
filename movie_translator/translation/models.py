from pathlib import Path
from typing import TypedDict

REPO_ROOT = Path(__file__).parent.parent.parent


class ModelConfig(TypedDict, total=False):
    huggingface_id: str
    description: str
    max_length: int


TRANSLATION_MODELS: dict[str, ModelConfig] = {
    'allegro': {
        'huggingface_id': 'allegro/BiDi-eng-pol',
        'description': 'Allegro BiDi English-Polish',
        'max_length': 512,
    },
    'nllb-600m': {
        'huggingface_id': 'facebook/nllb-200-distilled-600M',
        'description': 'NLLB-200 Distilled 600M (200 languages)',
        'max_length': 512,
    },
    'nllb-1.3b': {
        'huggingface_id': 'facebook/nllb-200-distilled-1.3B',
        'description': 'NLLB-200 Distilled 1.3B (200 languages)',
        'max_length': 512,
    },
}


def get_local_model_path(model_key: str) -> Path | None:
    """Return local model path if it exists, otherwise None."""
    full_path = REPO_ROOT / 'models' / model_key
    if full_path.exists() and (full_path / 'config.json').exists():
        return full_path
    return None


DEFAULT_MODEL = 'allegro'
DEFAULT_DEVICE = 'mps'
DEFAULT_BATCH_SIZE = 16
