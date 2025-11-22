import logging
from pathlib import Path
from typing import List, TYPE_CHECKING, cast
import gc
import os

if TYPE_CHECKING:
    from tqdm.std import tqdm as TqdmType

from tqdm.std import tqdm

from movie_translator.constants import DEFAULT_MODEL, LANGUAGE_POLISH, DEFAULT_BATCH_SIZE
from movie_translator.exceptions import TranslationError

logger = logging.getLogger(__name__)


class LocalLLMProvider:
    def __init__(self, device: str = "auto", model_name: str = DEFAULT_MODEL):
        self.device = self._get_device(device)
        self.model = None
        self.tokenizer = None
        self.model_name = model_name

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

        logger.info(f"Loading model {self.model_name} on {self.device}...")

        try:
            cache_root = Path(__file__).resolve().parent / "models"
            cache_root.mkdir(parents=True, exist_ok=True)
            model_cache_dir = cache_root / self.model_name.replace("/", "__")
            model_cache_dir.mkdir(parents=True, exist_ok=True)

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, cache_dir=str(model_cache_dir)
            )
            self.model = MarianMTModel.from_pretrained(
                self.model_name, cache_dir=str(model_cache_dir)
            )

            if self.device != "cpu":
                self.model = self.model.to(self.device)

            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise TranslationError(f"Failed to load model: {e}") from e

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
                max_length=128,
            )

            if self.device != "cpu":
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

            # Constrain generation length to avoid pathological long outputs that
            # can cause large memory spikes on CPU.
            generate_kwargs = {
                "max_new_tokens": 128,
                "early_stopping": True,
            }

            with torch.inference_mode():
                translations = self.model.generate(**encoded, **generate_kwargs)

            decoded = self.tokenizer.batch_decode(
                translations,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            del encoded
            del translations

            return decoded
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise TranslationError(f"Batch translation failed: {e}") from e


class FlanT5ENPLProvider:
    """Provider specialized for sdadas/flan-t5-base-translator-en-pl (T5-based EN->PL)."""

    def __init__(self, device: str = "auto", model_name: str = DEFAULT_MODEL):
        self.device = self._get_device(device)
        self.model = None
        self.tokenizer = None
        self.model_name = model_name

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
            from transformers import T5ForConditionalGeneration, T5Tokenizer
        except ImportError:
            raise TranslationError(
                "transformers library not installed. Install with: uv add transformers torch"
            )

        logger.info(f"Loading model {self.model_name} on {self.device}...")

        try:
            cache_root = Path(__file__).resolve().parent / "models"
            cache_root.mkdir(parents=True, exist_ok=True)
            model_cache_dir = cache_root / self.model_name.replace("/", "__")
            model_cache_dir.mkdir(parents=True, exist_ok=True)

            self.tokenizer = T5Tokenizer.from_pretrained(
                self.model_name, cache_dir=str(model_cache_dir)
            )
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.model_name, cache_dir=str(model_cache_dir)
            )

            if self.device != "cpu":
                self.model = self.model.to(self.device)

            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise TranslationError(f"Failed to load model: {e}") from e

    def translate_batch(self, texts: List[str], target_lang: str = "pol") -> List[str]:
        self.load_model()

        if not texts:
            return []

        try:
            import torch
        except ImportError:
            raise TranslationError("torch not installed")

        # T5 uses a task prefix; for translation we can use "translate English to Polish: "
        prefix = "translate English to Polish: "
        prepared_texts = [prefix + text for text in texts]

        try:
            encoded = self.tokenizer.batch_encode_plus(
                prepared_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )

            if self.device != "cpu":
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

            # Constrain generation length to avoid pathological long outputs.
            generate_kwargs = {
                "max_new_tokens": 128,
                "early_stopping": True,
            }

            with torch.inference_mode():
                translations = self.model.generate(**encoded, **generate_kwargs)

            decoded = self.tokenizer.batch_decode(
                translations,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            del encoded
            del translations

            return decoded
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise TranslationError(f"Batch translation failed: {e}") from e


class HFInferenceProvider:
    """Provider that uses Hugging Face hosted inference API for translation.

    This offloads the model to Hugging Face's infrastructure and requires
    an HF_TOKEN environment variable to be set.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        try:
            from huggingface_hub import InferenceClient
        except ImportError as e:
            raise TranslationError(
                "huggingface_hub not installed. Install with: uv add huggingface_hub"
            ) from e

        api_key = os.getenv("HF_TOKEN")
        if not api_key:
            raise TranslationError("HF_TOKEN environment variable is not set")

        self.model_name = model_name
        self.client = InferenceClient(provider="hf-inference", api_key=api_key)

    def translate_batch(self, texts: List[str], target_lang: str = "pol") -> List[str]:
        if not texts:
            return []

        results: List[str] = []
        for text in texts:
            prepared = f">>{target_lang}<< {text}"
            try:
                out = self.client.translation(
                    prepared,
                    model=self.model_name,
                )
            except Exception as e:
                logger.error(f"HF inference translation failed: {e}")
                raise TranslationError(f"HF inference translation failed: {e}") from e

            # The client returns a string for translation() calls.
            results.append(str(out))

        return results

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
                max_length=128,
            )

            if self.device != "cpu":
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

            # Constrain generation length to avoid pathological long outputs that
            # can cause large memory spikes on CPU.
            generate_kwargs = {
                "max_new_tokens": 128,
                "early_stopping": True,
            }

            with torch.inference_mode():
                translations = self.model.generate(**encoded, **generate_kwargs)

            decoded = self.tokenizer.batch_decode(
                translations,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            del encoded
            del translations

            return decoded
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise TranslationError(f"Batch translation failed: {e}") from e


def translate_file_local(
    input_path: str,
    output_path: str,
    device: str = "auto",
    model_name: str = DEFAULT_MODEL,
    target_language: str = LANGUAGE_POLISH,
    batch_size: int = DEFAULT_BATCH_SIZE,
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

    # Choose backend: HF hosted inference or local models.
    if device == "hf":
        provider = HFInferenceProvider(model_name=model_name)
    else:
        # Use the T5-based provider for the default EN->PL model.
        if model_name == DEFAULT_MODEL:
            provider = FlanT5ENPLProvider(device=device, model_name=model_name)
        else:
            provider = LocalLLMProvider(device=device, model_name=model_name)
        provider.load_model()

    lang_code_map = {
        "Polish": "pol",
        "English": "eng",
        "Czech": "ces",
        "Slovak": "slk",
    }
    target_lang_code = lang_code_map.get(target_language, "pol")

    total_lines = len(subs)

    logger.info(f"Translating {total_lines} subtitle lines...")

    pbar = cast(
        "TqdmType", tqdm(total=total_lines, desc="Translating", unit="line")
    )

    for i in range(0, total_lines, batch_size):
        batch_events = subs[i : i + batch_size]
        batch_texts = [event.text for event in batch_events]
        translated_batch = provider.translate_batch(batch_texts, target_lang_code)

        for event, translation in zip(batch_events, translated_batch):
            event.text = translation

        del batch_texts
        del translated_batch

        if i and i % (batch_size * 50) == 0:
            gc.collect()

        pbar.update(len(batch_events))

    pbar.close()

    logger.info(f"Saving translation to {output_path}")
    try:
        subs.save(output_path)
    except Exception as e:
        logger.error(f"Failed to save translation: {e}")
        raise TranslationError(f"Failed to save translation to {output_path}") from e
