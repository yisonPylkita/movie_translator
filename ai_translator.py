#!/usr/bin/env python3
"""
AI Translation System for Subtitles
Based on the working setup from previous session
Memory-optimized version
"""

import gc
import time

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DEFAULT_MODEL = 'allegro/BiDi-eng-pol'
DEFAULT_DEVICE = 'mps'  # Apple Silicon GPU
DEFAULT_BATCH_SIZE = 16  # Optimized for MacBook memory


class SubtitleTranslator:
    """AI-powered subtitle translator using BiDi-eng-pol model with memory optimization."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = DEFAULT_DEVICE,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        self.model_name = model_name
        self.device = self._determine_device(device)
        self.batch_size = batch_size
        self.translation_pipeline = None
        self.tokenizer = None
        self.model = None

        print(
            f'🤖 Initializing AI Translator: {model_name} on {self.device} with batch size {batch_size}'
        )

    def _determine_device(self, device: str) -> str:
        """Determine the best device for MacBook (MPS optimized)."""
        if device == 'mps':
            return 'mps'
        else:
            # Any other value defaults to CPU fallback
            return 'cpu'

    def _clear_memory(self):
        """Clear memory caches for MacBook (MPS optimized)."""
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

    def load_model(self):
        """Load the translation model with memory optimization."""
        print('📥 Loading translation model...')

        try:
            self._clear_memory()

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device != 'cpu' else torch.float32,
                low_cpu_mem_usage=True,
            )

            self.model.to(self.device)

            self.translation_pipeline = None

            print(f'   ✅ Model loaded successfully on {self.device}')
            return True

        except Exception as e:
            print(f'   ❌ Failed to load model: {e}')
            return False

    def translate_texts(self, texts: list[str], progress_callback=None) -> list[str]:
        """Translate a list of texts using direct model calls with optional progress callback."""
        print(f'🔄 Translating {len(texts)} texts...')

        if not texts:
            return []

        translations = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        start_time = time.time()

        # Process batches with optional progress callback
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1

            try:
                batch_translations = self._translate_batch_direct(batch_texts)
                translations.extend(batch_translations)

                if progress_callback:
                    elapsed = time.time() - start_time
                    lines_processed = min(batch_num * self.batch_size, len(texts))
                    rate = lines_processed / elapsed if elapsed > 0 else 0
                    progress_callback(batch_num, total_batches, rate)

                if i > 0 and i % (self.batch_size * 50) == 0:
                    self._clear_memory()

            except Exception as e:
                if progress_callback:
                    progress_callback(batch_num, total_batches, 0, error=str(e)[:50])
                translations.extend(batch_texts)

        self._clear_memory()
        print(f'   ✅ Translation complete: {len(translations)} texts processed')

        return translations

    def _translate_batch_direct(self, texts: list[str]) -> list[str]:
        """Direct batch translation using model.generate() (proven working approach)."""
        try:
            import torch
        except ImportError as err:
            raise Exception('torch not installed') from err

        if 'bidi' in self.model_name.lower():
            target = 'pol'
            texts = [f'>>{target}<< {text}' for text in texts]

        encoded = self.tokenizer.batch_encode_plus(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
        )

        if self.device != 'cpu':
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

        return decoded

    def cleanup(self):
        """Clean up model and free memory."""
        print('🧹 Cleaning up AI Translator...')

        self.translation_pipeline = None

        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer

        self.translation_pipeline = None
        self.model = None
        self.tokenizer = None

        self._clear_memory()

    def translate_dialogue_lines(
        self, dialogue_lines: list[tuple[int, int, str]]
    ) -> list[tuple[int, int, str]]:
        """Translate dialogue lines with timing preservation and memory optimization."""
        if not dialogue_lines:
            return []

        texts = [text for _, _, text in dialogue_lines]

        translated_texts = self.translate_texts(texts)

        translated_lines = []
        for (start, end, _), translated_text in zip(dialogue_lines, translated_texts, strict=True):
            translated_lines.append((start, end, translated_text))

        return translated_lines


def test_translation():
    """Test the translation system with sample texts."""
    print('🧪 Testing AI Translation System')
    print('=' * 50)

    translator = SubtitleTranslator()
    if translator.load_model():
        test_texts = [
            'Hello, how are you?',
            'This is a test sentence.',
            'The weather is nice today.',
        ]

        results = translator.translate_texts(test_texts)

        print('\nTranslation Results:')
        for original, translated in zip(test_texts, results, strict=True):
            print(f'  EN: {original}')
            print(f'  PL: {translated}')
            print()

        translator.cleanup()
    else:
        print('Failed to load model for testing')


if __name__ == '__main__':
    test_translation()
