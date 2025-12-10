from pathlib import Path
from typing import TypedDict

REPO_ROOT = Path(__file__).parent.parent.parent


class ModelConfig(TypedDict, total=False):
    name: str
    local_path: str
    description: str
    max_length: int


TRANSLATION_MODELS: dict[str, ModelConfig] = {
    'allegro': {
        'name': 'allegro/BiDi-eng-pol',
        'local_path': 'models/allegro',
        'description': 'Allegro BiDi English-Polish',
        'max_length': 512,
    },
}


def get_local_model_path(model_config: ModelConfig) -> Path | None:
    """Return local model path if it exists, otherwise None."""
    local_path = model_config.get('local_path')
    if not local_path:
        return None
    full_path = REPO_ROOT / local_path
    if full_path.exists() and (full_path / 'config.json').exists():
        return full_path
    return None


DEFAULT_MODEL = 'allegro'
DEFAULT_DEVICE = 'mps'
DEFAULT_BATCH_SIZE = 16
