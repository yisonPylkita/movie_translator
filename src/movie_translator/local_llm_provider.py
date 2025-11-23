import logging
import re
from pathlib import Path
from typing import List, TYPE_CHECKING, cast
import gc

if TYPE_CHECKING:
    from tqdm.std import tqdm as TqdmType

from tqdm.std import tqdm

from movie_translator.constants import DEFAULT_MODEL, DEFAULT_BATCH_SIZE
from movie_translator.exceptions import TranslationError

logger = logging.getLogger(__name__)

# Pattern for stripping ALL formatting tags (HTML and SSA-style)
# Matches: <tag>, </tag>, {\\tag}, {\\tag1}, etc.
FORMATTING_TAG_PATTERN = re.compile(r'<[^>]+>|{\\[^}]+}')


class TranslationProvider:
    def __init__(self, device: str = "auto", model_name: str = DEFAULT_MODEL):
        self.device = self._get_device(device)
        self.model = None
        self.tokenizer = None
        self.model_name = model_name
        self.use_fp16 = self.device == "mps"

    def _get_device(self, device: str) -> str:
        if device != "auto":
            return device

        try:
            import torch

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.info("Detected M1/M2/M3 Mac - using MPS acceleration")
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass

        logger.info("Falling back to CPU")
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
                self.model_name, cache_dir=str(model_cache_dir), local_files_only=False
            )

            self.model = MarianMTModel.from_pretrained(
                self.model_name, cache_dir=str(model_cache_dir), local_files_only=False
            )

            if self.device == "mps":
                self.model = self.model.to(self.device)
                self.model = self.model.half()
                logger.info("M1 optimization: Using float16 precision for 2x speed")
            elif self.device == "cuda":
                self.model = self.model.to(self.device)
                self.model = self.model.half()

            self.model.eval()
            logger.info(f"Model loaded successfully on {self.device}")
            logger.info(f"Using precision: {'float16' if self.use_fp16 else 'float32'}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise TranslationError(f"Failed to load model: {e}") from e

    def translate_batch(self, texts: List[str]) -> List[str]:
        self.load_model()

        if not texts:
            return []

        try:
            import torch
        except ImportError:
            raise TranslationError("torch not installed")

        try:
            if "bidi" in self.model_name.lower():
                model_parts = self.model_name.split("-")
                if len(model_parts) >= 3:
                    target = "pol"
                    texts = [f">>{target}<< {text}" for text in texts]
                    logger.debug(f"BiDi model detected, using prefix: >>{target}<<")

            encoded = self.tokenizer.batch_encode_plus(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

            if self.device != "cpu":
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.inference_mode():
                translations = self.model.generate(
                    **encoded,
                    max_new_tokens=128,
                    num_beams=1,
                    early_stopping=True,
                    do_sample=False,
                )

            decoded = self.tokenizer.batch_decode(
                translations,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            del encoded
            del translations
            if self.device == "mps":
                torch.mps.empty_cache()

            return decoded

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise TranslationError(f"Batch translation failed: {e}") from e


def strip_html_tags(text: str) -> str:
    """Remove all formatting tags from subtitle text.

    Removes both HTML tags (<i>, <b>, etc.) and SSA-style formatting ({\\i1}, {\\b1}, etc.)
    used internally by pysubs2.

    Args:
        text: Input text with potential formatting tags

    Returns:
        Text with all formatting tags removed
    """
    return FORMATTING_TAG_PATTERN.sub('', text)


def translate_file(
    input_path: str,
    output_path: str,
    device: str = "auto",
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    try:
        import pysubs2
    except ImportError:
        raise TranslationError("pysubs2 not installed")

    logger.info(f"Loading subtitles from {input_path}")
    subs = pysubs2.load(input_path)

    if not subs:
        raise TranslationError("No subtitles found in file")

    provider = TranslationProvider(device=device, model_name=model_name)
    provider.load_model()

    total_lines = len(subs)
    logger.info(
        f"Translating {total_lines} subtitle lines in batches of {batch_size}..."
    )
    logger.info(
        f"Device: {provider.device}, Precision: {'float16' if provider.use_fp16 else 'float32'}"
    )

    pbar = cast("TqdmType", tqdm(total=total_lines, desc="Translating", unit="lines"))

    for i in range(0, total_lines, batch_size):
        batch_events = subs[i : i + batch_size]

        # Always strip formatting tags from source text before translation
        # Preserve line break structure by translating line-by-line
        batch_texts = []
        line_break_structures = []

        for event in batch_events:
            clean_text = strip_html_tags(event.text)
            # Split by \N (pysubs2 line break marker)
            lines = clean_text.split('\\N')
            line_break_structures.append(len(lines))
            # Add each line separately for translation
            batch_texts.extend(lines)

        translated_batch = provider.translate_batch(batch_texts)

        # Reconstruct with original line break structure
        translated_idx = 0
        for event, num_lines in zip(batch_events, line_break_structures):
            # Get the translated lines for this event
            translated_lines = translated_batch[translated_idx:translated_idx + num_lines]
            translated_idx += num_lines

            # Join with line breaks and strip any remaining tags
            translation = '\\N'.join(translated_lines)
            translation = strip_html_tags(translation)
            event.text = translation

        del batch_texts
        del translated_batch

        if i > 0 and i % (batch_size * 50) == 0:
            gc.collect()

        pbar.update(len(batch_events))

    pbar.close()

    logger.info(f"Saving translation to {output_path}")
    output_ext = Path(output_path).suffix.lower()

    if output_ext == ".ass":
        subs.save(output_path, format_="ass")
    elif output_ext == ".srt":
        subs.save(output_path, format_="srt")
    else:
        subs.save(output_path)

    logger.info(f"Translation complete! Saved to {output_path}")


translate_file_local = translate_file
