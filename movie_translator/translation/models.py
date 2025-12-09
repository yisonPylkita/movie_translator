TRANSLATION_MODELS = {
    'allegro': {
        'name': 'allegro/BiDi-eng-pol',
        'description': 'Allegro BiDi English-Polish (default)',
        'max_length': 512,
    },
    'mbart': {
        'name': 'facebook/mbart-large-50-many-to-many-mmt',
        'description': 'mBART Many-to-Many Multilingual',
        'max_length': 512,
    },
}

DEFAULT_MODEL = 'allegro'
DEFAULT_DEVICE = 'mps'
DEFAULT_BATCH_SIZE = 16
