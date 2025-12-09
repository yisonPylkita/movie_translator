from typing import TypedDict


class ModelConfig(TypedDict, total=False):
    name: str
    description: str
    max_length: int
    use_slow_tokenizer: bool
    base_tokenizer: str


TRANSLATION_MODELS: dict[str, ModelConfig] = {
    'allegro': {
        'name': 'allegro/BiDi-eng-pol',
        'description': 'Allegro BiDi English-Polish',
        'max_length': 512,
    },
    'facebook': {
        'name': 'facebook/mbart-large-50-many-to-many-mmt',
        'description': 'Facebook mBART Multilingual',
        'max_length': 512,
    },
}

DEFAULT_MODEL = 'allegro'
DEFAULT_DEVICE = 'mps'
DEFAULT_BATCH_SIZE = 16
