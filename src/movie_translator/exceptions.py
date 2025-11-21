class MovieTranslatorError(Exception):
    pass


class SubtitleNotFoundError(MovieTranslatorError):
    pass


class TranslationError(MovieTranslatorError):
    pass


class MKVProcessingError(MovieTranslatorError):
    pass


class ConfigurationError(MovieTranslatorError):
    pass
