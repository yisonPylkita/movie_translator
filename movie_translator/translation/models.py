TRANSLATION_MODELS = {
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
