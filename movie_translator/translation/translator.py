import gc
import time
from collections.abc import Callable

import torch
from rich.progress import Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ..logging import console, logger
from ..types import DialogueLine
from .enhancements import PreprocessingStats, postprocess_translation, preprocess_for_translation
from .models import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_MODEL,
    TRANSLATION_MODELS,
    ModelConfig,
    get_local_model_path,
)

# Callback receives (batch_num, total_batches, lines_per_second)
ProgressCallback = Callable[[int, int, float], None]


class SubtitleTranslator:
    def __init__(
        self,
        model_key: str = DEFAULT_MODEL,
        device: str = DEFAULT_DEVICE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        enable_enhancements: bool = True,
    ):
        self.model_key = model_key
        self.model_config = self._get_model_config(model_key)
        self.model_path = self._resolve_model_path()
        self.device = 'mps' if device == 'mps' else 'cpu'
        self.batch_size = batch_size
        self.enable_enhancements = enable_enhancements
        self.preprocessing_stats = PreprocessingStats()
        self.tokenizer = None
        self.model = None

        logger.info(f'Initializing translator on {self.device}')
        if enable_enhancements:
            logger.info('Translation enhancements enabled (idioms, short phrases, cleanup)')

    def _resolve_model_path(self) -> str:
        """Return local model path if available, otherwise HuggingFace model ID."""
        local_path = get_local_model_path(self.model_key)
        if local_path:
            return str(local_path)
        return self.model_config.get('huggingface_id', '')

    def _get_model_config(self, model_key: str) -> ModelConfig:
        if model_key in TRANSLATION_MODELS:
            return TRANSLATION_MODELS[model_key]
        return {
            'huggingface_id': model_key,
            'description': 'Custom model',
            'max_length': 512,
        }

    def _clear_memory(self):
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

    def load_model(self) -> bool:
        logger.info('Loading model...')

        try:
            self._clear_memory()
            self._load_tokenizer()
            self._load_model()
            return True
        except Exception as e:
            logger.error(f'Failed to load model: {e}')
            return False

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def _load_model(self):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device != 'cpu' else torch.float32,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)

    def translate_texts(
        self, texts: list[str], progress_callback: ProgressCallback | None = None
    ) -> list[str]:
        if not texts:
            return []

        translations = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        start_time = time.time()

        for i in range(0, len(texts), self.batch_size):
            batch_num = i // self.batch_size + 1
            batch_texts = texts[i : i + self.batch_size]

            batch_translations = self._translate_batch(batch_texts)
            translations.extend(batch_translations)

            if progress_callback:
                elapsed = time.time() - start_time
                lines_processed = min(batch_num * self.batch_size, len(texts))
                rate = lines_processed / elapsed if elapsed > 0 else 0
                progress_callback(batch_num, total_batches, rate)

            self._periodic_memory_cleanup(i)

        self._clear_memory()

        # Log preprocessing statistics if enhancements are enabled
        if self.enable_enhancements and self.preprocessing_stats.total_processed > 0:
            logger.info(self.preprocessing_stats.get_summary())

        return translations

    def _periodic_memory_cleanup(self, index: int):
        if index > 0 and index % (self.batch_size * 50) == 0:
            self._clear_memory()

    def _translate_batch(self, texts: list[str]) -> list[str]:
        self._validate_inputs(texts)

        if self.enable_enhancements:
            enhanced_texts, skip_indices, cached_translations = self._apply_preprocessing(texts)
        else:
            enhanced_texts = texts
            skip_indices = set()
            cached_translations = {}

        processed_texts = self._preprocess_texts(enhanced_texts)
        encoded = self._encode_texts(processed_texts)
        outputs = self._generate_translations(encoded)
        decoded = self._decode_outputs(outputs)

        del encoded
        del outputs

        if self.enable_enhancements:
            decoded = self._apply_postprocessing(decoded)

        return self._apply_fallbacks(texts, decoded, skip_indices, cached_translations)

    def _validate_inputs(self, texts: list[str]) -> None:
        """Log warnings for inputs that may produce poor translations."""
        for i, text in enumerate(texts):
            stripped = text.strip()
            if len(stripped) < 3:
                logger.warning(
                    f'Very short input at index {i}: "{text}" ({len(stripped)} chars) - '
                    'may produce empty or poor translation'
                )
            elif len(stripped) < 5:
                logger.debug(
                    f'Short input at index {i}: "{text}" ({len(stripped)} chars) - '
                    'translation quality may vary'
                )

    def _apply_preprocessing(self, texts: list[str]) -> tuple[list[str], set[int], dict[int, str]]:
        """Apply preprocessing enhancements to texts."""
        if not self.enable_enhancements:
            return texts, set(), {}

        processed_texts = []
        skip_indices = set()
        cached_translations = {}

        for i, text in enumerate(texts):
            processed, was_mapped = preprocess_for_translation(text, self.preprocessing_stats)
            processed_texts.append(processed)

            if was_mapped:
                skip_indices.add(i)
                cached_translations[i] = processed

        return processed_texts, skip_indices, cached_translations

    def _apply_postprocessing(self, translations: list[str]) -> list[str]:
        """Apply postprocessing cleanup to translations."""
        return [postprocess_translation(t) for t in translations]

    def _apply_fallbacks(
        self,
        originals: list[str],
        translations: list[str],
        skip_indices: set[int] | None = None,
        cached_translations: dict[int, str] | None = None,
    ) -> list[str]:
        """Apply fallback logic for empty or invalid translations."""
        if skip_indices is None:
            skip_indices = set()
        if cached_translations is None:
            cached_translations = {}

        result = []
        for i, (original, translated) in enumerate(zip(originals, translations, strict=True)):
            if i in skip_indices:
                result.append(cached_translations.get(i, translated))
                continue

            stripped_translation = translated.strip()

            if not stripped_translation:
                logger.warning(
                    f'Empty translation for line {i}: "{original}" - '
                    'using original text as fallback'
                )
                result.append(original)
            elif len(stripped_translation) < 2 and len(original.strip()) > 5:
                logger.warning(
                    f'Suspiciously short translation for line {i}: '
                    f'"{original}" -> "{translated}" - using original as fallback'
                )
                result.append(original)
            else:
                result.append(translated)

        return result

    def _preprocess_texts(self, texts: list[str]) -> list[str]:
        huggingface_id = self.model_config.get('huggingface_id', '').lower()
        if 'bidi' in huggingface_id:
            return [f'>>pol<< {text}' for text in texts]
        return texts

    def _encode_texts(self, texts: list[str]) -> dict:
        assert self.tokenizer is not None
        encoded = self.tokenizer.batch_encode_plus(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.model_config.get('max_length', 512),
        )
        if self.device != 'cpu':
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
        return encoded

    def _generate_translations(self, encoded: dict) -> torch.Tensor:
        assert self.model is not None
        with torch.inference_mode():
            return self.model.generate(
                **encoded,
                max_new_tokens=128,
                num_beams=1,
                early_stopping=True,
                do_sample=False,
            )

    def _decode_outputs(self, outputs: torch.Tensor) -> list[str]:
        assert self.tokenizer is not None
        return self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    def cleanup(self):
        logger.info('🧹 Cleaning up AI Translator...')
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        self.model = None
        self.tokenizer = None
        self._clear_memory()


def translate_dialogue_lines(
    dialogue_lines: list[DialogueLine],
    device: str,
    batch_size: int,
    model: str,
) -> list[DialogueLine]:
    translator = SubtitleTranslator(device=device, batch_size=batch_size, model_key=model)

    if not translator.load_model():
        return []

    texts = [line.text for line in dialogue_lines]
    total_batches = (len(texts) + batch_size - 1) // batch_size

    with Progress(
        SpinnerColumn(),
        TextColumn('[progress.description]{task.description}'),
        TaskProgressColumn(),
        TextColumn('•'),
        TimeElapsedColumn(),
        TextColumn('•'),
        TextColumn('{task.fields[rate]}'),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(
            f'[cyan]Translating {len(texts)} lines...[/cyan]',
            total=total_batches,
            rate='',
        )

        def on_progress(batch_num: int, total_batches: int, rate: float) -> None:
            progress.update(task, advance=1, rate=f'{rate:.1f}/s')

        translated_texts = translator.translate_texts(texts, on_progress)

    translator.cleanup()
    gc.collect()

    return [
        DialogueLine(line.start_ms, line.end_ms, text)
        for line, text in zip(dialogue_lines, translated_texts, strict=True)
    ]
