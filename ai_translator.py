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

# Available translation models with their configurations
TRANSLATION_MODELS = {
    'allegro': {
        'name': 'allegro/BiDi-eng-pol',
        'description': 'Allegro BiDi English-Polish (default)',
        'type': 'seq2seq',
        'max_length': 512,
    },
    'flan-t5': {
        'name': 'sdadas/flan-t5-base-translator-en-pl',
        'description': 'FLAN-T5 English-Polish Translator',
        'type': 'seq2seq',
        'max_length': 512,
        'base_tokenizer': 'google/flan-t5-base',
        'use_slow_tokenizer': True,
    },
    'mbart': {
        'name': 'facebook/mbart-large-50-many-to-many-mmt',
        'description': 'mBART Many-to-Many Multilingual',
        'type': 'seq2seq',
        'max_length': 512,
    },
    'nllb': {
        'name': 'facebook/nllb-200-distilled-600M',
        'description': 'NLLB-200 Distilled 600M (200 languages)',
        'type': 'seq2seq',
        'max_length': 512,
        'src_lang': 'eng_Latn',
        'tgt_lang': 'pol_Latn',
    },
}

DEFAULT_MODEL = 'allegro'  # Use model key instead of full name
DEFAULT_DEVICE = 'mps'  # Apple Silicon GPU
DEFAULT_BATCH_SIZE = 16  # Optimized for MacBook memory


class SubtitleTranslator:
    """AI-powered subtitle translator with multiple model support and memory optimization."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = DEFAULT_DEVICE,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        # Support both model keys and full model names
        if model_name in TRANSLATION_MODELS:
            self.model_config = TRANSLATION_MODELS[model_name]
        else:
            # Assume it's a full model name
            self.model_config = {
                'name': model_name,
                'description': 'Custom model',
                'type': 'seq2seq',
                'max_length': 512,
            }

        self.model_name = self.model_config['name']
        self.device = self._determine_device(device)
        self.batch_size = batch_size
        self.translation_pipeline = None
        self.tokenizer = None
        self.model = None

        print(
            f'🤖 Initializing AI Translator: {self.model_config["description"]} on {self.device} with batch size {batch_size}'
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

            # Handle tokenizer loading with base model support for FLAN-T5
            if self.model_config.get('use_slow_tokenizer') and self.model_config.get(
                'base_tokenizer'
            ):
                # Use base model tokenizer in slow mode (for FLAN-T5 compatibility)
                base_tokenizer = self.model_config['base_tokenizer']
                self.tokenizer = AutoTokenizer.from_pretrained(base_tokenizer, use_fast=False)
                print(f'   - Using base tokenizer: {base_tokenizer}')
            else:
                # Standard tokenizer loading
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Use AutoModelForSeq2SeqLM for all available models
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                dtype=torch.float16 if self.device != 'cpu' else torch.float32,
                low_cpu_mem_usage=True,
            )

            self.model.to(self.device)

            self.translation_pipeline = None

            print(f'   ✅ Model loaded successfully on {self.device}')
            return True

        except Exception as e:
            print(f'   ❌ Failed to load model: {e}')
            import traceback

            traceback.print_exc()
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
        """Direct batch translation using model.generate() with model-specific handling."""
        try:
            import torch
        except ImportError as err:
            raise Exception('torch not installed') from err

        # Handle different model types - simplified for available models
        if 'bidi' in self.model_name.lower():
            # BiDi models need language prefix
            target = 'pol'
            texts = [f'>>{target}<< {text}' for text in texts]
        elif 'flan-t5' in self.model_name.lower():
            # FLAN-T5 models work with direct text input
            # No special prefix needed
            pass
        elif 'mbart' in self.model_name.lower():
            # mBART needs language codes set on tokenizer
            # We'll handle this in the model.generate call
            pass
        elif 'nllb' in self.model_name.lower():
            # NLLB works with direct text input
            # Language codes handled in generation
            pass
        # Other models work with direct text input (no special handling needed)

        encoded = self.tokenizer.batch_encode_plus(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.model_config.get('max_length', 512),
        )

        if self.device != 'cpu':
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.inference_mode():
            # Handle different model types for generation
            if 'mbart' in self.model_name.lower():
                # Set source language and force Polish target
                self.tokenizer.src_lang = 'en_XX'
                translations = self.model.generate(
                    **encoded,
                    forced_bos_token_id=self.tokenizer.lang_code_to_id['pl_PL'],
                    max_new_tokens=128,
                    num_beams=1,
                    early_stopping=True,
                    do_sample=False,
                )
            elif 'nllb' in self.model_name.lower():
                # NLLB uses convert_tokens_to_ids for language codes
                tgt_lang = self.model_config.get('tgt_lang', 'pol_Latn')
                forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
                translations = self.model.generate(
                    **encoded,
                    forced_bos_token_id=forced_bos_token_id,
                    max_new_tokens=128,
                    num_beams=1,
                    early_stopping=True,
                    do_sample=False,
                )
            else:
                # Standard generation for other models (Allegro, FLAN-T5)
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
