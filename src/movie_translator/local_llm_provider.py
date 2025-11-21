import logging
from pathlib import Path
from typing import Optional, List, TYPE_CHECKING, cast

if TYPE_CHECKING:
    from tqdm.std import tqdm as TqdmType

from tqdm.std import tqdm

from movie_translator.constants import BIDI_MODEL_NAME, LANGUAGE_POLISH
from movie_translator.exceptions import TranslationError

logger = logging.getLogger(__name__)


class LocalLLMProvider:
    def __init__(self, device: str = "auto"):
        self.device = self._get_device(device)
        self.model = None
        self.tokenizer = None
        self.model_name = BIDI_MODEL_NAME

    def _get_device(self, device: str) -> str:
        if device != "auto":
            return device

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def load_model(self):
        if self.model is not None:
            return

        try:
            from transformers import AutoTokenizer, MarianMTModel
        except ImportError:
            raise TranslationError(
                "transformers library not installed. Install with: uv add transformers torch"
            )

        logger.info(f"Loading BiDi model {self.model_name} on {self.device}...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = MarianMTModel.from_pretrained(self.model_name)

            if self.device != "cpu":
                self.model = self.model.to(self.device)

            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise TranslationError(f"Failed to load BiDi model: {e}") from e

    def translate_batch(self, texts: List[str], target_lang: str = "pol") -> List[str]:
        self.load_model()

        if not texts:
            return []

        try:
            import torch
        except ImportError:
            raise TranslationError("torch not installed")

        prepared_texts = [f">>{target_lang}<< {text}" for text in texts]

        try:
            encoded = self.tokenizer.batch_encode_plus(
                prepared_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

            if self.device != "cpu":
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                translations = self.model.generate(**encoded)

            decoded = self.tokenizer.batch_decode(
                translations,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            return decoded
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise TranslationError(f"Batch translation failed: {e}") from e


def translate_file_local(
    input_path: str,
    output_path: str,
    device: str = "auto",
    target_language: str = LANGUAGE_POLISH,
) -> None:
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input subtitle file not found: {input_path}")

    try:
        import pysubs2
    except ImportError:
        raise TranslationError("pysubs2 not installed")

    logger.info(f"Loading subtitles from {input_path}")
    subs = pysubs2.load(input_path)

    if not subs:
        raise TranslationError("No subtitles found in file")

    provider = LocalLLMProvider(device=device)
    provider.load_model()

    lang_code_map = {
        "Polish": "pol",
        "English": "eng",
        "Czech": "ces",
        "Slovak": "slk",
    }
    target_lang_code = lang_code_map.get(target_language, "pol")

    texts_to_translate = [event.text for event in subs]

    logger.info(f"Translating {len(texts_to_translate)} subtitle lines...")

    batch_size = 32
    translated_texts = []

    pbar = cast(
        "TqdmType", tqdm(total=len(texts_to_translate), desc="Translating", unit="line")
    )

    for i in range(0, len(texts_to_translate), batch_size):
        batch = texts_to_translate[i : i + batch_size]
        translated_batch = provider.translate_batch(batch, target_lang_code)
        translated_texts.extend(translated_batch)
        pbar.update(len(batch))

    pbar.close()

    for event, translation in zip(subs, translated_texts):
        event.text = translation

    logger.info(f"Saving translation to {output_path}")
    try:
        subs.save(output_path)
    except Exception as e:
        logger.error(f"Failed to save translation: {e}")
        raise TranslationError(f"Failed to save translation to {output_path}") from e
