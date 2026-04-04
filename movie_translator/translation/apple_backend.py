"""Apple Translation backend — on-device EN→PL via macOS Translation framework."""

from __future__ import annotations

import json
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path

from ..logging import logger
from ..types import ProgressCallback
from .enhancements import (
    PreprocessingStats,
    extract_placeholders,
    postprocess_translation,
    preprocess_for_translation,
    restore_placeholders,
)
from .sentence_merger import merge_for_translation, unmerge_translations

# Path to Swift source and compiled binary
_SWIFT_DIR = Path(__file__).parent / 'swift'
_SWIFT_SOURCE = _SWIFT_DIR / 'translate_bridge.swift'
_SWIFT_BINARY = _SWIFT_DIR / 'translate_bridge'

# A line that is nothing but a placeholder tag + punctuation should skip
# the model — it would hallucinate random text for meaningless input.
_PLACEHOLDER_ONLY_RE = re.compile(r'^__\w+__[.!?,;:\u2026\s]*$')


class AppleTranslationError(RuntimeError):
    """Raised when Apple Translation fails."""

    def __init__(self, message: str, code: str = 'internal'):
        super().__init__(message)
        self.code = code


def is_available() -> bool:
    """Check if Apple Translation backend can be used on this system.

    Requirements:
    - macOS 26.0+ (Tahoe)
    - Swift source file exists
    """
    if platform.system() != 'Darwin':
        return False

    mac_ver = platform.mac_ver()[0]
    if not mac_ver:
        return False
    try:
        major = int(mac_ver.split('.')[0])
        if major < 26:
            return False
    except ValueError, IndexError:
        return False

    if not _SWIFT_SOURCE.exists():
        return False

    return True


def check_languages_installed() -> bool:
    """Quick check if EN→PL translation works by doing a test translation."""
    try:
        binary = _ensure_binary()
        _call_swift_binary(binary, ['test'], timeout=15)
        return True
    except AppleTranslationError, subprocess.TimeoutExpired, FileNotFoundError:
        return False


class AppleTranslationBackend:
    """On-device translation via Apple's Translation framework.

    Calls a compiled Swift CLI binary via subprocess, passing texts
    as JSON on stdin and reading translations from stdout.

    The binary is auto-compiled from source on first use. Translation
    languages must be pre-downloaded via System Settings.
    """

    def __init__(self, batch_size: int = 200, enable_enhancements: bool = True):
        self.batch_size = batch_size
        self.enable_enhancements = enable_enhancements
        self.preprocessing_stats = PreprocessingStats()
        self.proper_nouns: set[str] = set()
        self._binary_path = _ensure_binary()
        logger.info('Apple Translation backend ready')

    def translate_texts(
        self,
        texts: list[str],
        progress_callback: ProgressCallback | None = None,
    ) -> list[str]:
        """Translate English texts to Polish.

        Applies sentence merging, enhancements, and fallbacks — same
        contract as SubtitleTranslator.translate_texts().
        """
        if not texts:
            return []

        merged_texts, groups = merge_for_translation(texts)
        logger.debug(
            f'Sentence merging: {len(texts)} lines \u2192 {len(merged_texts)} translation units'
        )

        # Apply enhancements: placeholders + preprocessing
        placeholder_mappings: list[dict[str, str]] = []
        skip_indices: set[int] = set()
        cached_translations: dict[int, str] = {}

        if self.enable_enhancements:
            processed_texts = []
            for i, text in enumerate(merged_texts):
                protected, mapping = extract_placeholders(
                    text, self.preprocessing_stats, proper_nouns=self.proper_nouns or None
                )
                placeholder_mappings.append(mapping)

                # If the line is nothing but a placeholder tag + punctuation
                # (e.g. "__NM0__..." from "Lord Boscone..."), skip the model.
                if _PLACEHOLDER_ONLY_RE.match(protected.strip()):
                    restored = restore_placeholders(protected, mapping)
                    processed_texts.append(restored)
                    skip_indices.add(i)
                    cached_translations[i] = restored
                    continue

                processed, was_mapped = preprocess_for_translation(
                    protected, self.preprocessing_stats
                )
                processed_texts.append(processed)
                if was_mapped:
                    skip_indices.add(i)
                    cached_translations[i] = processed
            merged_texts = processed_texts
        else:
            placeholder_mappings = [{} for _ in merged_texts]

        # Collect texts that need actual translation (not cached)
        texts_to_translate: list[tuple[int, str]] = []
        for i, text in enumerate(merged_texts):
            if i not in skip_indices:
                texts_to_translate.append((i, text))

        translations = [''] * len(merged_texts)

        # Fill in cached translations first
        for i, cached in cached_translations.items():
            translations[i] = cached

        total_lines = len(texts)
        start_time = time.time()
        translate_texts_only = [t for _, t in texts_to_translate]

        for batch_start in range(0, len(translate_texts_only), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(translate_texts_only))
            batch = translate_texts_only[batch_start:batch_end]

            batch_results = _call_swift_binary(self._binary_path, batch)

            for j, translated in enumerate(batch_results):
                original_idx = texts_to_translate[batch_start + j][0]
                translations[original_idx] = translated

            if progress_callback:
                units_done = batch_end + len(skip_indices)
                lines_done = min(
                    sum(len(g.line_indices) for g in groups[:units_done]),
                    total_lines,
                )
                elapsed = time.time() - start_time
                rate = lines_done / elapsed if elapsed > 0 else 0
                progress_callback(lines_done, total_lines, rate)

        if self.enable_enhancements:
            translations = [postprocess_translation(t) for t in translations]

        translations = [
            restore_placeholders(text, mapping)
            for text, mapping in zip(translations, placeholder_mappings, strict=True)
        ]

        translations = _apply_fallbacks(
            merged_texts, translations, skip_indices, cached_translations
        )

        if self.enable_enhancements and self.preprocessing_stats.total_processed > 0:
            logger.info(self.preprocessing_stats.get_summary())

        return unmerge_translations(translations, groups, texts)

    def cleanup(self) -> None:
        """No-op — Apple backend is stateless."""


def _apply_fallbacks(
    originals: list[str],
    translations: list[str],
    skip_indices: set[int],
    cached_translations: dict[int, str],
) -> list[str]:
    """Apply fallback logic: empty/suspicious translations -> original text."""
    result = []
    for i, (original, translated) in enumerate(zip(originals, translations, strict=True)):
        if i in skip_indices:
            result.append(cached_translations.get(i, translated))
            continue

        stripped = translated.strip()
        if not stripped:
            logger.warning(f'Empty Apple translation for: "{original}" \u2014 using original')
            result.append(original)
        elif len(stripped) < 2 and len(original.strip()) > 5:
            logger.warning(
                f'Suspiciously short Apple translation: "{original}" \u2192 "{translated}" \u2014 using original'
            )
            result.append(original)
        else:
            result.append(translated)

    return result


def _ensure_binary() -> Path:
    """Compile the Swift bridge binary if it doesn't exist or source is newer."""
    if not _SWIFT_SOURCE.exists():
        raise FileNotFoundError(
            f'Swift bridge source not found: {_SWIFT_SOURCE}\n'
            'Ensure the movie_translator package is installed correctly.'
        )

    needs_compile = (
        not _SWIFT_BINARY.exists() or _SWIFT_SOURCE.stat().st_mtime > _SWIFT_BINARY.stat().st_mtime
    )

    if needs_compile:
        logger.info('Compiling Apple Translation bridge...')
        swiftc = shutil.which('swiftc')
        if swiftc is None:
            raise FileNotFoundError(
                'Swift compiler (swiftc) not found. '
                'Install Xcode or Command Line Tools: xcode-select --install'
            )

        result = subprocess.run(
            [
                swiftc,
                '-parse-as-library',
                '-O',
                '-framework',
                'Translation',
                str(_SWIFT_SOURCE),
                '-o',
                str(_SWIFT_BINARY),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            raise RuntimeError(f'Failed to compile Apple Translation bridge:\n{result.stderr}')

        logger.info(f'Compiled: {_SWIFT_BINARY}')

    return _SWIFT_BINARY


def _call_swift_binary(binary: Path, texts: list[str], timeout: int = 120) -> list[str]:
    """Call the Swift bridge binary with a batch of texts.

    Returns list of translated texts (same length as input).
    Raises AppleTranslationError on failure.
    """
    request = json.dumps({'texts': texts, 'source': 'en', 'target': 'pl'})

    result = subprocess.run(
        [str(binary)],
        input=request.encode('utf-8'),
        capture_output=True,
        timeout=timeout,
    )

    if not result.stdout:
        stderr = result.stderr.decode('utf-8', errors='replace')
        raise AppleTranslationError(
            f'Swift bridge returned no output (exit code {result.returncode}): {stderr}'
        )

    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise AppleTranslationError(
            f'Invalid JSON from Swift bridge: {e}\nOutput: {result.stdout[:200]}'
        ) from e

    if 'error' in response and response['error']:
        raise AppleTranslationError(
            response['error'],
            code=response.get('code', 'internal'),
        )

    translations = response.get('translations', [])
    if len(translations) != len(texts):
        raise AppleTranslationError(
            f'Translation count mismatch: sent {len(texts)}, got {len(translations)}'
        )

    return translations
