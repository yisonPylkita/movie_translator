import gc
import time
from typing import Protocol

import torch
from rich.progress import Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ..logging import console, logger
from ..types import DialogueLine
from .models import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_MODEL,
    TRANSLATION_MODELS,
    ModelConfig,
    get_local_model_path,
)


class ProgressCallback(Protocol):
    def __call__(
        self,
        batch_num: int,
        total_batches: int,
        lines_per_second: float,
        error: str | None,
    ) -> None: ...


class SubtitleTranslator:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = DEFAULT_DEVICE,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        self.model_config = self._get_model_config(model_name)
        self.model_path = self._resolve_model_path()
        self.device = 'mps' if device == 'mps' else 'cpu'
        self.batch_size = batch_size
        self.tokenizer = None
        self.model = None

        logger.info(f'Initializing translator on {self.device}')

    def _resolve_model_path(self) -> str:
        """Return local model path if available, otherwise HuggingFace model ID."""
        local_path = get_local_model_path(self.model_config)
        if local_path:
            return str(local_path)
        return self.model_config.get('name', '')

    def _get_model_config(self, model_name: str) -> ModelConfig:
        if model_name in TRANSLATION_MODELS:
            return TRANSLATION_MODELS[model_name]
        return {
            'name': model_name,
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

    def translate_texts(self, texts: list[str], progress_callback: ProgressCallback) -> list[str]:
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
            self._report_progress(
                progress_callback, batch_num, total_batches, start_time, len(texts)
            )
            self._periodic_memory_cleanup(i)

        self._clear_memory()
        return translations

    def _report_progress(
        self,
        callback: ProgressCallback,
        batch_num: int,
        total_batches: int,
        start_time: float,
        total_texts: int,
    ):
        elapsed = time.time() - start_time
        lines_processed = min(batch_num * self.batch_size, total_texts)
        rate = lines_processed / elapsed if elapsed > 0 else 0
        callback(batch_num, total_batches, rate, None)

    def _periodic_memory_cleanup(self, index: int):
        if index > 0 and index % (self.batch_size * 50) == 0:
            self._clear_memory()

    def _translate_batch(self, texts: list[str]) -> list[str]:
        processed_texts = self._preprocess_texts(texts)
        encoded = self._encode_texts(processed_texts)
        outputs = self._generate_translations(encoded)
        decoded = self._decode_outputs(outputs)

        del encoded
        del outputs
        return decoded

    def _preprocess_texts(self, texts: list[str]) -> list[str]:
        if 'bidi' in self.model_path.lower():
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
    translator = SubtitleTranslator(device=device, batch_size=batch_size, model_name=model)

    if not translator.load_model():
        return []

    texts = [text for _, _, text in dialogue_lines]
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

        def on_progress(
            batch_num: int,
            total_batches: int,
            lines_per_second: float,
            error: str | None,
        ) -> None:
            if error:
                progress.update(task, advance=1, rate=f'[red]{error}[/red]')
            else:
                progress.update(task, advance=1, rate=f'{lines_per_second:.1f}/s')

        translated_texts = translator.translate_texts(texts, on_progress)

    translator.cleanup()
    gc.collect()

    return [
        DialogueLine(line.start_ms, line.end_ms, text)
        for line, text in zip(dialogue_lines, translated_texts, strict=True)
    ]
