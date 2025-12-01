#!/usr/bin/env python3
"""
AI Translation System for Subtitles
Based on the working setup from previous session
Memory-optimized version
"""

import gc
import os
import time

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Model configuration (from previous working setup)
DEFAULT_MODEL = "allegro/BiDi-eng-pol"
DEFAULT_DEVICE = "auto"  # Will use MPS for Apple Silicon
DEFAULT_BATCH_SIZE = 16  # Restored from working old implementation



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
        """Clear memory caches and force garbage collection (proven approach)."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Add MPS support like the working old implementation
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

    
    def load_model(self):
        """Load the translation model with memory optimization."""
        print("📥 Loading translation model...")

        try:
            # Clear memory before loading
            self._clear_memory()

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

            print(f"   ✅ Model loaded successfully on {self.device}")
            return True

        except Exception as e:
            print(f"   ❌ Failed to load model: {e}")
            return False

    def translate_texts(self, texts: list[str], progress_callback=None) -> list[str]:
        """Translate a list of texts using direct model calls with optional progress callback."""
        print(f"🔄 Translating {len(texts)} texts...")

        if not texts:
            return []

        translations = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        start_time = time.time()
        
        # Process batches with optional progress callback
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            
            try:
                # Direct model translation (proven approach from old implementation)
                batch_translations = self._translate_batch_direct(batch_texts)
                translations.extend(batch_translations)
                
                # Update progress if callback provided
                if progress_callback:
                    elapsed = time.time() - start_time
                    lines_processed = min(batch_num * self.batch_size, len(texts))
                    rate = lines_processed / elapsed if elapsed > 0 else 0
                    progress_callback(batch_num, total_batches, rate)
                
                # Periodic cleanup like old implementation (every 50 batches worth of text)
                if i > 0 and i % (self.batch_size * 50) == 0:
                    self._clear_memory()
                    
            except Exception as e:
                # Show error on progress if callback provided
                if progress_callback:
                    progress_callback(batch_num, total_batches, 0, error=str(e)[:50])
                # Fallback: return original texts for this batch
                translations.extend(batch_texts)

        # Final cleanup
        self._clear_memory()
        print(f"   ✅ Translation complete: {len(translations)} texts processed")

        return translations

    def _translate_batch_direct(self, texts: list[str]) -> list[str]:
        """Direct batch translation using model.generate() (proven working approach)."""
        try:
            import torch
        except ImportError:
            raise Exception("torch not installed")

        # Handle BiDi model prefix like old implementation
        if "bidi" in self.model_name.lower():
            target = "pol"
            texts = [f">>{target}<< {text}" for text in texts]

        # Tokenize batch
        encoded = self.tokenizer.batch_encode_plus(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        # Move to device if not CPU
        if self.device != "cpu":
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

        # Generate translations
        with torch.inference_mode():
            translations = self.model.generate(
                **encoded,
                max_new_tokens=128,
                num_beams=1,
                early_stopping=True,
                do_sample=False,
            )

        # Decode results
        decoded = self.tokenizer.batch_decode(
            translations,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # Clean up tensors
        del encoded
        del translations
        
        return decoded

    def cleanup(self):
        """Clean up model and free memory."""
        print("🧹 Cleaning up AI Translator...")

        # Clear pipeline reference if it exists
        self.translation_pipeline = None

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
