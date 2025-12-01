#!/usr/bin/env python3
"""
AI Translation System for Subtitles
Based on the working setup from previous session
Memory-optimized version
"""

import gc
import os

import psutil
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Model configuration (from previous working setup)
DEFAULT_MODEL = "allegro/BiDi-eng-pol"
DEFAULT_DEVICE = "auto"  # Will use MPS for Apple Silicon
DEFAULT_BATCH_SIZE = 8  # Reduced from 16 for memory efficiency


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

        print("🤖 Initializing AI Translator (Memory Optimized):")
        print(f"   - Model: {model_name}")
        print(f"   - Device: {self.device}")
        print(f"   - Batch Size: {batch_size} (reduced for memory)")

    def _determine_device(self, device: str) -> str:
        """Determine the best device for translation."""
        if device == "auto":
            if torch.backends.mps.is_available():
                return "mps"  # Apple Silicon
            elif torch.cuda.is_available():
                return "cuda"  # NVIDIA
            else:
                return "cpu"
        return device

    def _clear_memory(self):
        """Clear memory caches and force garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _get_memory_info(self) -> str:
        """Get current memory usage information."""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                cached = torch.cuda.memory_reserved() / 1024**3
                return f"Process: {memory_mb:.1f}MB, GPU: {allocated:.1f}GB allocated, {cached:.1f}GB cached"
            else:
                return f"Process: {memory_mb:.1f}MB"
        except Exception:
            return "Memory info unavailable"

    def load_model(self):
        """Load the translation model with memory optimization."""
        print("📥 Loading translation model (Memory Optimized)...")

        try:
            # Clear memory before loading
            self._clear_memory()
            print(f"   - Memory before loading: {self._get_memory_info()}")

            # Load tokenizer and model with memory optimization
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Load model with reduced precision if possible
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                low_cpu_mem_usage=True
            )

            # Move model to device
            self.model.to(self.device)

            # NOTE: NOT creating persistent pipeline to avoid memory leaks
            # We'll create fresh pipelines for each text
            self.translation_pipeline = None

            print(f"   - Memory after loading: {self._get_memory_info()}")
            print(f"   ✅ Model loaded successfully on {self.device}")
            return True

        except Exception as e:
            print(f"   ❌ Failed to load model: {e}")
            return False

    def translate_texts(self, texts: list[str]) -> list[str]:
        """Translate a list of texts with aggressive memory management."""
        print(f"🔄 Translating {len(texts)} texts (Aggressive Memory Management)...")
        print(f"   - Memory before translation: {self._get_memory_info()}")

        translations = []

        # VERY AGGRESSIVE: Process 1 text at a time but with shared model
        # This is slower but should prevent any memory accumulation

        for i, text in enumerate(texts):
            batch_num = i + 1

            # Memory check every 10 texts
            if i % 10 == 0:
                print(f"   📦 Text {batch_num}/{len(texts)} (Memory: {self._get_memory_info()})")

            try:
                # Create fresh pipeline for EACH text (but reuse model/tokenizer)
                temp_pipeline = pipeline(
                    "translation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    clean_up_tokenization_spaces=True,
                )

                # Translate single text
                result = temp_pipeline(text)
                translated_text = result[0]["translation_text"]
                translations.append(translated_text)

                # AGGRESSIVE cleanup
                del temp_pipeline

                # Clear any model state that might be accumulating
                if hasattr(self.model, 'cache'):
                    self.model.cache = None
                if hasattr(self.model, 'past_key_values'):
                    self.model.past_key_values = None

                # Force garbage collection
                self._clear_memory()
                gc.collect()

                # Memory check every 10 texts
                if i % 10 == 0:
                    print(f"      🧹 Text {batch_num} cleaned (Memory: {self._get_memory_info()})")

            except Exception as e:
                print(f"      ❌ Text {batch_num} failed: {e}")
                # Fallback: return original text
                translations.append(text)

        # Final memory cleanup
        self._clear_memory()
        print(f"   - Memory after translation: {self._get_memory_info()}")
        print(f"   ✅ Translation complete: {len(translations)} texts processed")

        return translations

    def cleanup(self):
        """Clean up model and free memory."""
        print("🧹 Cleaning up AI Translator...")

        # Clear pipeline if it exists
        if self.translation_pipeline:
            del self.translation_pipeline

        # Clear model and tokenizer
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer

        # Set all to None
        self.translation_pipeline = None
        self.model = None
        self.tokenizer = None

        # Final memory cleanup
        self._clear_memory()
        print(f"   - Memory after cleanup: {self._get_memory_info()}")

    def translate_dialogue_lines(
        self, dialogue_lines: list[tuple[int, int, str]]
    ) -> list[tuple[int, int, str]]:
        """Translate dialogue lines with timing preservation and memory optimization."""
        if not dialogue_lines:
            return []

        # Extract just the text for translation
        texts = [text for _, _, text in dialogue_lines]

        # Translate texts
        translated_texts = self.translate_texts(texts)

        # Reconstruct with timing
        translated_lines = []
        for (start, end, _), translated_text in zip(dialogue_lines, translated_texts):
            translated_lines.append((start, end, translated_text))

        return translated_lines


def test_translation():
    """Test the translation system with sample texts."""
    print("🧪 Testing AI Translation System")
    print("=" * 50)

    translator = SubtitleTranslator()
    if translator.load_model():
        test_texts = [
            "Hello, how are you?",
            "This is a test sentence.",
            "The weather is nice today.",
        ]

        results = translator.translate_texts(test_texts)

        print("\nTranslation Results:")
        for original, translated in zip(test_texts, results):
            print(f"  EN: {original}")
            print(f"  PL: {translated}")
            print()

        translator.cleanup()
    else:
        print("Failed to load model for testing")


if __name__ == "__main__":
    test_translation()
