import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING, cast

if TYPE_CHECKING:
    from tqdm.std import tqdm as TqdmType

from tqdm.std import tqdm

from PySubtrans.Options import Options
from PySubtrans.Subtitles import Subtitles
from PySubtrans.SubtitleBatcher import SubtitleBatcher
from PySubtrans.SubtitleTranslator import SubtitleTranslator
from PySubtrans.Providers.Provider_OpenAI import OpenAiProvider

from movie_translator.constants import (
    DEFAULT_MODEL,
    DEFAULT_BATCH_SIZE,
    DEFAULT_SCENE_THRESHOLD,
    LANGUAGE_POLISH,
    PROVIDER_OPENAI,
    PROVIDER_LOCAL,
    DEFAULT_DEVICE,
)
from movie_translator.exceptions import TranslationError

logger = logging.getLogger(__name__)


def translate_file(
    input_path: str,
    output_path: str,
    provider: str = PROVIDER_OPENAI,
    api_key: Optional[str] = None,
    target_language: str = LANGUAGE_POLISH,
    movie_name: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    scene_threshold: float = DEFAULT_SCENE_THRESHOLD,
    device: str = DEFAULT_DEVICE,
) -> None:
    if provider == PROVIDER_LOCAL:
        from movie_translator.local_llm_provider import translate_file_local

        translate_file_local(
            input_path=input_path,
            output_path=output_path,
            device=device,
            target_language=target_language,
        )
        return

    if not api_key:
        raise TranslationError("API key is required for OpenAI provider")
    settings_dict = {
        "api_key": api_key,
        "target_language": target_language,
        "movie_name": movie_name,
        "model": model,
        "min_batch_size": batch_size,
        "max_batch_size": batch_size,
        "scene_threshold": scene_threshold,
        "provider": "OpenAI",
    }

    options = Options(settings_dict)

    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input subtitle file not found: {input_path}")

    logger.info(f"Loading subtitles from {input_path}")
    subtitles = Subtitles(filepath=input_path, outputpath=output_path, settings=options)
    try:
        subtitles.LoadSubtitles()
    except Exception as e:
        logger.error(f"Failed to load subtitles: {e}")
        raise TranslationError(f"Failed to load subtitles from {input_path}") from e

    if not subtitles.originals:
        error_msg = f"No subtitles found in file: {input_path}"
        logger.warning(error_msg)
        raise TranslationError(error_msg)

    logger.info("Batching subtitles into scenes...")
    batcher = SubtitleBatcher(options)
    try:
        subtitles.scenes = batcher.BatchSubtitles(subtitles.originals)
    except Exception as e:
        logger.error(f"Failed to batch subtitles: {e}")
        raise TranslationError("Failed to batch subtitles into scenes") from e

    logger.info(f"Created {len(subtitles.scenes)} scenes.")

    openai_provider = OpenAiProvider(options)
    if not openai_provider.ValidateSettings():
        error_msg = getattr(openai_provider, "validation_message", "Unknown error")
        raise TranslationError(f"Invalid provider settings: {error_msg}")

    translator = SubtitleTranslator(options, openai_provider)

    total_batches = sum(len(scene.batches) for scene in subtitles.scenes)
    pbar = cast("TqdmType", tqdm(total=total_batches, desc="Translating", unit="batch"))

    def on_batch_translated(sender, batch, **kwargs):
        pbar.update(1)

    def on_error(sender, message, **kwargs):
        logger.error(f"Translation Error: {message}")

    def on_warning(sender, message, **kwargs):
        logger.warning(f"Translation Warning: {message}")

    translator.events.batch_translated.connect(on_batch_translated)
    translator.events.error.connect(on_error)
    translator.events.warning.connect(on_warning)

    try:
        translator.TranslateSubtitles(subtitles)
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise TranslationError("Subtitle translation failed") from e
    finally:
        pbar.close()

    if subtitles.translated:
        logger.info(f"Saving translation to {output_path}")
        try:
            subtitles.SaveTranslation(output_path)
        except Exception as e:
            logger.error(f"Failed to save translation: {e}")
            raise TranslationError(
                f"Failed to save translation to {output_path}"
            ) from e
    else:
        error_msg = "No translations were generated"
        logger.error(error_msg)
        raise TranslationError(error_msg)
