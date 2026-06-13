from pathlib import Path
from typing import TypedDict

REPO_ROOT = Path(__file__).parent.parent.parent


class ModelConfig(TypedDict, total=False):
    huggingface_id: str
    description: str
    max_length: int


# Only the seq2seq backend we actually ship lives here. Apple Translation is a
# separate non-seq2seq backend handled in apple_backend.py. MLX is a
# separate non-PyTorch backend handled in mlx_backend.py.
TRANSLATION_MODELS: dict[str, ModelConfig] = {
    'allegro': {
        'huggingface_id': 'allegro/BiDi-eng-pol',
        'description': 'Allegro BiDi English-Polish (PyTorch)',
        'max_length': 512,
    },
    'mlx': {
        'huggingface_id': 'allegro/BiDi-eng-pol',
        'description': 'Allegro BiDi English-Polish (MLX, Metal-native)',
        'max_length': 512,
    },
}


def get_local_model_path(model_key: str) -> Path | None:
    """Return local model path if it exists, otherwise None."""
    full_path = REPO_ROOT / 'models' / model_key
    if full_path.exists() and (full_path / 'config.json').exists():
        return full_path
    return None


# Default model priority:
# 1. 'apple' (macOS Translation framework) — fastest, zero memory, macOS 26+
# 2. 'mlx' (Metal-native INT8) — fallback on any Apple Silicon
DEFAULT_MODEL = 'apple'
DEFAULT_DEVICE = 'mps'
DEFAULT_BATCH_SIZE = 4
