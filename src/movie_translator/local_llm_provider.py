import logging
from pathlib import Path
from typing import List, TYPE_CHECKING, cast, Optional
import gc

if TYPE_CHECKING:
    from tqdm.std import tqdm as TqdmType

from tqdm.std import tqdm

from movie_translator.constants import DEFAULT_MODEL, LANGUAGE_POLISH, DEFAULT_BATCH_SIZE
from movie_translator.exceptions import TranslationError

logger = logging.getLogger(__name__)


class OptimizedT5Provider:
    """Optimized T5 provider for M1 MacBook Air.

    Specifically for T5-based models like sdadas/flan-t5-base-translator-en-pl.
    Same optimizations as Marian but with T5-specific handling.
    """

    def __init__(self, device: str = "auto", model_name: str = DEFAULT_MODEL):
        self.device = self._get_device(device)
        self.model = None
        self.tokenizer = None
        self.model_name = model_name
        self.use_fp16 = self.device == "mps"  # Use float16 on M1 for speed

    def _get_device(self, device: str) -> str:
        """Detect best device, prioritizing MPS for M1 Macs."""
        if device != "auto":
            return device

        try:
            import torch

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.info("Detected M1/M2 Mac - using MPS acceleration")
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass

        logger.info("Falling back to CPU")
        return "cpu"

    def load_model(self):
        """Load T5 model with M1 optimizations."""
        if self.model is not None:
            return

        try:
            from transformers import T5ForConditionalGeneration, AutoTokenizer
            import torch
        except ImportError:
            raise TranslationError(
                "transformers library not installed. Install with: uv add transformers torch"
            )

        logger.info(f"Loading T5 model {self.model_name} on {self.device}...")

        try:
            cache_root = Path(__file__).resolve().parent / "models"
            cache_root.mkdir(parents=True, exist_ok=True)
            model_cache_dir = cache_root / self.model_name.replace("/", "__")
            model_cache_dir.mkdir(parents=True, exist_ok=True)

            # Use AutoTokenizer with Python 3.14 workaround
            # Python 3.14 has a bug with sentencepiece - use legacy=False to avoid it
            import os
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=str(model_cache_dir),
                local_files_only=False,
                legacy=False  # Avoid Python 3.14 sentencepiece bug
            )

            self.model = T5ForConditionalGeneration.from_pretrained(
                self.model_name,
                cache_dir=str(model_cache_dir),
                local_files_only=False
            )

            # M1 optimization: move to MPS and use float16
            if self.device == "mps":
                self.model = self.model.to(self.device)
                self.model = self.model.half()
                logger.info("M1 optimization: Using float16 precision for 2x speed")
            elif self.device == "cuda":
                self.model = self.model.to(self.device)
                self.model = self.model.half()
            elif self.device == "cpu":
                pass  # CPU stays in float32

            self.model.eval()
            logger.info(f"T5 model loaded successfully on {self.device}")
            logger.info(f"Using precision: {'float16' if self.use_fp16 else 'float32'}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise TranslationError(f"Failed to load model: {e}") from e

    def translate_batch(self, texts: List[str], target_lang: Optional[str] = None) -> List[str]:
        """Translate a batch of texts using T5."""
        self.load_model()

        if not texts:
            return []

        try:
            import torch
        except ImportError:
            raise TranslationError("torch not installed")

        try:
            # T5 models use task prefix
            prefix = "translate English to Polish: "
            prepared_texts = [prefix + text for text in texts]

            encoded = self.tokenizer.batch_encode_plus(
                prepared_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )

            if self.device != "cpu":
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

            # Speed optimizations
            generate_kwargs = {
                "max_new_tokens": 128,
                "num_beams": 1,  # Greedy decoding for speed
                "early_stopping": True,
                "do_sample": False,
            }

            with torch.inference_mode():
                translations = self.model.generate(**encoded, **generate_kwargs)

            decoded = self.tokenizer.batch_decode(
                translations,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            # Memory cleanup
            del encoded
            del translations
            if self.device == "mps":
                torch.mps.empty_cache()

            return decoded

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise TranslationError(f"Batch translation failed: {e}") from e


class OptimizedMarianProvider:
    """Optimized Marian MT provider for M1 MacBook Air.

    Optimizations:
    - Uses Helsinki-NLP/opus-mt-en-pl (78MB, faster than flan-t5)
    - MPS acceleration for M1 (Metal Performance Shaders)
    - Float16 precision for 2x speed boost on M1
    - Larger batch sizes (16+) for better throughput
    - Aggressive memory management
    """

    def __init__(self, device: str = "auto", model_name: str = DEFAULT_MODEL):
        self.device = self._get_device(device)
        self.model = None
        self.tokenizer = None
        self.model_name = model_name
        self.use_fp16 = self.device == "mps"  # Use float16 on M1 for speed

    def _get_device(self, device: str) -> str:
        """Detect best device, prioritizing MPS for M1 Macs."""
        if device != "auto":
            return device

        try:
            import torch

            # Prioritize MPS for M1/M2 Macs (fastest on Apple Silicon)
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.info("Detected M1/M2 Mac - using MPS acceleration")
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass

        logger.info("Falling back to CPU")
        return "cpu"

    def load_model(self):
        """Load model with M1 optimizations."""
        if self.model is not None:
            return

        try:
            from transformers import AutoTokenizer, MarianMTModel
            import torch
        except ImportError:
            raise TranslationError(
                "transformers library not installed. Install with: uv add transformers torch"
            )

        logger.info(f"Loading model {self.model_name} on {self.device}...")

        try:
            # Cache model locally to avoid re-downloading
            cache_root = Path(__file__).resolve().parent / "models"
            cache_root.mkdir(parents=True, exist_ok=True)
            model_cache_dir = cache_root / self.model_name.replace("/", "__")
            model_cache_dir.mkdir(parents=True, exist_ok=True)

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=str(model_cache_dir),
                local_files_only=False
            )

            # Load model
            self.model = MarianMTModel.from_pretrained(
                self.model_name,
                cache_dir=str(model_cache_dir),
                local_files_only=False
            )

            # Move to device and optimize
            if self.device == "mps":
                # M1 optimization: use float16 for 2x speed
                self.model = self.model.to(self.device)
                self.model = self.model.half()  # Convert to float16
                logger.info("M1 optimization: Using float16 precision for 2x speed")
            elif self.device == "cuda":
                self.model = self.model.to(self.device)
                self.model = self.model.half()  # Also use fp16 on CUDA
            elif self.device == "cpu":
                # CPU stays in float32
                pass

            # Set to eval mode for inference
            self.model.eval()

            logger.info(f"Model loaded successfully on {self.device}")
            logger.info(f"Using precision: {'float16' if self.use_fp16 else 'float32'}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise TranslationError(f"Failed to load model: {e}") from e

    def translate_batch(self, texts: List[str], target_lang: Optional[str] = None) -> List[str]:
        """Translate a batch of texts.

        Args:
            texts: List of English texts to translate
            target_lang: Target language code (not used for Marian models)

        Returns:
            List of translated texts
        """
        self.load_model()

        if not texts:
            return []

        try:
            import torch
        except ImportError:
            raise TranslationError("torch not installed")

        try:
            # Marian models don't need language prefix
            # They're trained for specific language pairs
            encoded = self.tokenizer.batch_encode_plus(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,  # Keep inputs short for speed
            )

            # Move to device
            if self.device != "cpu":
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

            # Generate translations with speed optimizations
            generate_kwargs = {
                "max_new_tokens": 128,  # Limit output length
                "num_beams": 1,  # Greedy decoding (fastest)
                "early_stopping": True,
                "do_sample": False,  # No sampling for speed
            }

            with torch.inference_mode():  # Faster than no_grad()
                translations = self.model.generate(**encoded, **generate_kwargs)

            # Decode translations
            decoded = self.tokenizer.batch_decode(
                translations,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            # Aggressive memory cleanup
            del encoded
            del translations
            if self.device == "mps":
                torch.mps.empty_cache()

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
    """Translate an SRT subtitle file using a local model.

    Args:
        input_path: Path to input SRT file
        output_path: Path to output SRT file
        device: Device to use (auto/cpu/mps/cuda)
        model_name: HuggingFace model name
        target_language: Target language name
        batch_size: Number of lines to translate per batch
    """
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

    # Use optimized provider based on model type
    # T5-based models (like sdadas/flan-t5-*) need different handling than Marian models
    if "t5" in model_name.lower():
        from movie_translator.local_llm_provider import OptimizedT5Provider
        provider = OptimizedT5Provider(device=device, model_name=model_name)
    else:
        provider = OptimizedMarianProvider(device=device, model_name=model_name)
    provider.load_model()

    total_lines = len(subs)
    logger.info(f"Translating {total_lines} subtitle lines in batches of {batch_size}...")
    logger.info(f"Device: {provider.device}, Precision: {'float16' if provider.use_fp16 else 'float32'}")

    # Progress bar
    pbar = cast(
        "TqdmType", tqdm(total=total_lines, desc="Translating", unit="lines")
    )

    # Translate in batches
    for i in range(0, total_lines, batch_size):
        batch_events = subs[i : i + batch_size]
        batch_texts = [event.text for event in batch_events]

        # Translate batch
        translated_batch = provider.translate_batch(batch_texts)

        # Update subtitles
        for event, translation in zip(batch_events, translated_batch):
            event.text = translation

        # Clean up memory every 50 batches
        del batch_texts
        del translated_batch

        if i > 0 and i % (batch_size * 50) == 0:
            gc.collect()

        pbar.update(len(batch_events))

    pbar.close()

    # Save translated subtitles
    logger.info(f"Saving translation to {output_path}")
    try:
        subs.save(output_path)
        logger.info(f"Translation complete! Saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save translation: {e}")
        raise TranslationError(f"Failed to save translation to {output_path}") from e
