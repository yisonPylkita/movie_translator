# Apple Translation Backend Design

## Goal

Add Apple's on-device Translation framework as a translation backend for English→Polish subtitle translation, callable from our Python codebase. This provides a free, fast, zero-dependency alternative to the current HuggingFace model backend — no API keys, no GPU required, no model downloads at runtime.

## Non-Goals

- Replacing the HuggingFace/Allegro backend (it remains the default)
- Supporting languages other than English→Polish
- Running on non-macOS platforms (Linux, Windows)
- Building a general-purpose translation framework

---

## Experiment Summary

### What We Tested

We built a proof-of-concept at `apple_translate_poc/` that translates ASS subtitle files from English to Polish using Apple's Translation framework (`import Translation`). The POC extracts subtitles from MKV files, parses ASS format, batch-translates dialogue lines, and writes translated `.ass` files.

### Key Findings

| Finding | Detail |
|---------|--------|
| **On-device** | Fully on-device after language download. No network needed, no API keys. |
| **EN→PL supported** | Polish was added to Apple Translate in 2022. Works reliably. |
| **Performance** | 297 subtitle lines in 6.3s (~47 lines/sec). 5 lines in 225ms via JSON batch. ~22ms/line at scale. |
| **macOS 26+ headless** | `TranslationSession(installedSource:target:)` works in a plain `async main()` — no SwiftUI needed. |
| **Batch API** | `session.translations(from:)` accepts `[Request]`, returns ordered `[Response]`. |
| **Quality** | ML-model level (comparable to Google Translate). Not LLM-level, but usable for subtitles. |
| **Language download** | Must be pre-downloaded via System Settings → General → Language & Region → Translation Languages. |

### Pitfalls Discovered

1. **SwiftUI + CLI args**: SwiftUI `WindowGroup` interprets command-line arguments as file paths to open, causing the app to hang indefinitely. Solved by using environment variables in the POC. **Not relevant for the headless approach** — the final implementation uses a plain `async main()` with no SwiftUI dependency.

2. **`LanguageAvailability.status()` unreliable**: Reports `.supported` even when models are on disk and functional. The session creation itself is the reliable check — it throws `TranslationError.notInstalled` if languages aren't ready.

3. **PyObjC can't bridge Swift APIs**: The Translation framework's public API is Swift-native. PyObjC only bridges Objective-C. The private ObjC classes (`_LTTranslationSession`) exist but are fragile and undocumented. Dead end.

4. **No Python package exists**: No `pyobjc-framework-Translation` or third-party wrapper. Subprocess with a compiled Swift binary is the only viable Python integration path.

5. **SwiftUI `.translationTask` callback + `session.translate()` deadlock**: When using the SwiftUI path (macOS 15), calling `session.translate()` from within the `.translationTask` callback hangs due to a MainActor deadlock. The headless macOS 26 path avoids this entirely.

### Translation Quality Samples

From the POC run on One Punch Man S01E01 (297 dialogue lines, 6.3s):

| English | Polish (Apple Translation) |
|---------|----------------------------|
| What on earth?! | Co u licha?! |
| Look over there! | Spójrz tam! |
| Just a guy who's a hero for fun. | Po prostu facet, który jest bohaterem dla zabawy. |
| I am Vaccine Man! | Jestem Człowiekiem Szczepionkowym! |
| The monster is wreaking destruction on an unprecedented scale! | Potwór sieje zniszczenia na bezprecedensową skalę! |
| Guess I'll go. | Chyba pójdę. |
| Your eyes are lifeless, just like mine. | Twoje oczy są martwe, tak jak moje. |

---

## Architecture

### Python ↔ Swift Bridge

The only viable path for Python integration is **subprocess with JSON I/O**:

```
Python (movie_translator)                 Swift CLI binary
─────────────────────────────────────────────────────────

translate_dialogue_lines()
  │
  ├─ Serialize texts to JSON ──stdin──►  Read JSON from stdin
  │                                       │
  │                                       ├─ Create TranslationSession
  │                                       ├─ Batch translate via
  │                                       │  session.translations(from:)
  │                                       ├─ Serialize results to JSON
  │                                       │
  ◄──────────────────────────stdout──────  Write JSON to stdout
  │
  └─ Deserialize, return translations
```

**JSON Protocol** (stdin/stdout):

```json
// Request (stdin)
{
  "texts": ["Hello world", "What on earth?!"],
  "source": "en",
  "target": "pl"
}

// Response (stdout) — success
{
  "translations": ["Witaj świecie", "Co u licha?!"],
  "elapsed_ms": 225
}

// Response (stdout) — error
{
  "error": "Translation languages not installed",
  "code": "not_installed"
}
```

Error codes: `"not_installed"` (languages not downloaded), `"unsupported"` (language pair not supported), `"internal"` (framework error).

### Swift CLI Binary

**File:** `movie_translator/translation/swift/translate_bridge.swift`

Single-file, no dependencies, no Package.swift needed. Compiled with:
```bash
swiftc -parse-as-library -O -framework Translation translate_bridge.swift -o translate_bridge
```

**Complete implementation:**

```swift
import Foundation
import Translation

struct BatchRequest: Codable {
    let texts: [String]
    let source: String
    let target: String
}

struct BatchResponse: Codable {
    let translations: [String]?
    let elapsed_ms: Int?
    let error: String?
    let code: String?
}

@main
struct TranslateBridge {
    static func main() async {
        do {
            let inputData = FileHandle.standardInput.readDataToEndOfFile()
            let request = try JSONDecoder().decode(BatchRequest.self, from: inputData)

            let session = TranslationSession(
                installedSource: Locale.Language(identifier: request.source),
                target: Locale.Language(identifier: request.target)
            )

            let start = CFAbsoluteTimeGetCurrent()
            let batchRequests = request.texts.enumerated().map { i, text in
                TranslationSession.Request(sourceText: text, clientIdentifier: "\(i)")
            }
            let responses = try await session.translations(from: batchRequests)
            let elapsedMs = Int((CFAbsoluteTimeGetCurrent() - start) * 1000)

            let response = BatchResponse(
                translations: responses.map(\.targetText),
                elapsed_ms: elapsedMs,
                error: nil,
                code: nil
            )
            let output = try JSONEncoder().encode(response)
            FileHandle.standardOutput.write(output)
            FileHandle.standardOutput.write("\n".data(using: .utf8)!)

        } catch let error as TranslationError {
            let code: String
            let message: String
            switch error {
            case .notInstalled:
                code = "not_installed"
                message = "Translation languages not installed. Download via: System Settings > General > Language & Region > Translation Languages"
            case .unsupportedLanguagePairing:
                code = "unsupported"
                message = "Unsupported language pairing"
            default:
                code = "internal"
                message = "Translation error: \(error)"
            }
            let errResponse = BatchResponse(translations: nil, elapsed_ms: nil, error: message, code: code)
            if let out = try? JSONEncoder().encode(errResponse) {
                FileHandle.standardOutput.write(out)
                FileHandle.standardOutput.write("\n".data(using: .utf8)!)
            }
            Foundation.exit(1)

        } catch {
            let errResponse = BatchResponse(translations: nil, elapsed_ms: nil, error: "\(error)", code: "internal")
            if let out = try? JSONEncoder().encode(errResponse) {
                FileHandle.standardOutput.write(out)
                FileHandle.standardOutput.write("\n".data(using: .utf8)!)
            }
            Foundation.exit(1)
        }
    }
}
```

**Key design decisions:**
- Errors are returned as JSON on stdout (not stderr) so Python can parse them uniformly
- Exit code 1 on error, 0 on success
- `TranslationError` cases are mapped to human-readable codes
- No SwiftUI dependency — runs headless on macOS 26+

---

## Existing Code: Integration Points

### Current Translation Flow

```
CLI (main.py)
  │  args.model = 'allegro' (choices=['allegro'])
  ▼
TranslationPipeline.__init__(model=args.model)
  │  stores in self.config = PipelineConfig(model=model)
  ▼
TranslateStage.run(ctx)
  │  reads ctx.config.model, ctx.config.device, ctx.config.batch_size
  ▼
translate_dialogue_lines(dialogue_lines, device, batch_size, model, progress_callback)
  │  calls _get_translator(device, batch_size, model) → cached SubtitleTranslator
  ▼
SubtitleTranslator.translate_texts(texts, progress_callback)
  │
  ├─ merge_for_translation(texts) → (merged_texts, groups)
  ├─ FOR EACH batch of merged_texts:
  │    ├─ extract_placeholders(text) → (protected_text, mapping)
  │    ├─ preprocess_for_translation(text) → (processed, was_mapped)
  │    ├─ _preprocess_texts() → add ">>pol<< " prefix for BiDi model
  │    ├─ tokenizer.batch_encode_plus() → encoded tensors
  │    ├─ model.generate() → output tensors
  │    ├─ tokenizer.batch_decode() → translated strings
  │    ├─ postprocess_translation() → cleanup
  │    ├─ restore_placeholders() → restore numbers/names
  │    └─ _apply_fallbacks() → use original if empty/suspicious
  ├─ unmerge_translations(translations, groups, texts)
  └─ return list[DialogueLine] with Polish text
```

### Files That Must Change

**`movie_translator/main.py`** — Add `'apple'` to CLI model choices:
```python
# Current (line ~45):
parser.add_argument('--model', choices=['allegro'], default='allegro')

# New:
parser.add_argument('--model', choices=['allegro', 'apple'], default='allegro')
```

**`movie_translator/translation/__init__.py`** — Route to correct backend:
```python
# Current:
from .translator import translate_dialogue_lines

# New:
from .translator import translate_dialogue_lines  # unchanged — routing happens inside
```

**`movie_translator/translation/translator.py`** — Add Apple backend dispatch in `translate_dialogue_lines()`:
```python
# Current function (lines 339-356):
def translate_dialogue_lines(dialogue_lines, device, batch_size, model, progress_callback=None):
    translator = _get_translator(device, batch_size, model)
    if translator is None:
        return []
    texts = [line.text for line in dialogue_lines]
    translated_texts = translator.translate_texts(texts, progress_callback)
    return [DialogueLine(l.start_ms, l.end_ms, t) for l, t in zip(dialogue_lines, translated_texts, strict=True)]

# New:
def translate_dialogue_lines(dialogue_lines, device, batch_size, model, progress_callback=None):
    if model == 'apple':
        from .apple_backend import AppleTranslationBackend
        backend = _get_apple_backend(batch_size)
    else:
        backend = _get_translator(device, batch_size, model)

    if backend is None:
        return []

    texts = [line.text for line in dialogue_lines]
    translated_texts = backend.translate_texts(texts, progress_callback)
    return [DialogueLine(l.start_ms, l.end_ms, t) for l, t in zip(dialogue_lines, translated_texts, strict=True)]
```

### Files That Stay Unchanged

- **`movie_translator/context.py`** — `PipelineConfig.model: str` already accepts arbitrary strings
- **`movie_translator/stages/translate.py`** — passes `ctx.config.model` through unchanged
- **`movie_translator/pipeline.py`** — passes `model=` through unchanged
- **`movie_translator/translation/sentence_merger.py`** — backend-agnostic text merging
- **`movie_translator/translation/enhancements.py`** — backend-agnostic preprocessing

### Key Interfaces to Implement Against

**Progress callback signature** (from `translator.py` line 33):
```python
ProgressCallback = Callable[[int, int, float], None]
# progress_callback(lines_done: int, total_lines: int, lines_per_second: float)
```

**Sentence merger** (from `sentence_merger.py`):
```python
def merge_for_translation(texts: list[str]) -> tuple[list[str], list[TranslationGroup]]:
    """Group texts and build merged translation inputs."""

def unmerge_translations(
    translated_texts: list[str],
    groups: list[TranslationGroup],
    original_texts: list[str],
) -> list[str]:
    """Split translated texts back to match original line count."""

@dataclass
class TranslationGroup:
    line_indices: list[int]        # indices into original texts list
    is_fragment_merge: bool        # True=space-joined fragments, False=||-separated sentences
```

**Enhancement functions** (from `enhancements.py`):
```python
def extract_placeholders(text: str, stats: PreprocessingStats | None = None) -> tuple[str, dict[str, str]]:
    """Replace numbers/URLs/names with __PHONE0__ etc. Returns (protected_text, mapping)."""

def restore_placeholders(text: str, mapping: dict[str, str]) -> str:
    """Restore __PHONE0__ back to original values."""

def preprocess_for_translation(text: str, stats: PreprocessingStats | None = None) -> tuple[str, bool]:
    """Returns (processed_text, was_direct_mapped). If was_direct_mapped=True, the processed_text IS the Polish translation (skip model)."""

def postprocess_translation(text: str) -> str:
    """Clean up model output: remove repetition, normalize punctuation, etc."""
```

**Fallback logic** (from `translator.py` lines 226-262):
- Empty translation → use original English text
- Translation < 2 chars when original > 5 chars → use original English text
- Skip indices from preprocessing → use cached Polish translation

---

## New Files: Complete Specifications

### `movie_translator/translation/swift/translate_bridge.swift`

See complete implementation in the "Swift CLI Binary" section above. This is the full, production-ready source.

### `movie_translator/translation/apple_backend.py`

```python
"""Apple Translation backend — on-device EN→PL via macOS Translation framework."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import time
from pathlib import Path

from ..logging import logger
from ..types import DialogueLine
from .enhancements import (
    PreprocessingStats,
    extract_placeholders,
    postprocess_translation,
    preprocess_for_translation,
    restore_placeholders,
)
from .sentence_merger import merge_for_translation, unmerge_translations
from .translator import ProgressCallback

# Path to Swift source and compiled binary
_SWIFT_DIR = Path(__file__).parent / 'swift'
_SWIFT_SOURCE = _SWIFT_DIR / 'translate_bridge.swift'
_SWIFT_BINARY = _SWIFT_DIR / 'translate_bridge'


class AppleTranslationError(RuntimeError):
    """Raised when Apple Translation fails."""

    def __init__(self, message: str, code: str = 'internal'):
        super().__init__(message)
        self.code = code


def is_available() -> bool:
    """Check if Apple Translation backend can be used on this system.

    Requirements:
    - macOS 26.0+ (Tahoe)
    - Swift compiler available (Xcode or Command Line Tools)
    - Swift source file exists
    """
    if platform.system() != 'Darwin':
        return False

    # Check macOS version — need 26.0+
    mac_ver = platform.mac_ver()[0]  # e.g. '26.4'
    if not mac_ver:
        return False
    try:
        major = int(mac_ver.split('.')[0])
        if major < 26:
            return False
    except (ValueError, IndexError):
        return False

    # Check Swift source exists
    if not _SWIFT_SOURCE.exists():
        return False

    return True


def check_languages_installed() -> bool:
    """Quick check if EN→PL translation works by doing a test translation.

    Returns True if languages are installed, False otherwise.
    Compiles the binary if needed.
    """
    try:
        binary = _ensure_binary()
        result = _call_swift_binary(binary, ['test'], timeout=15)
        return result is not None
    except (AppleTranslationError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


class AppleTranslationBackend:
    """On-device translation via Apple's Translation framework.

    Calls a compiled Swift CLI binary via subprocess, passing texts
    as JSON on stdin and reading translations from stdout.

    The binary is auto-compiled from source on first use. Translation
    languages must be pre-downloaded via System Settings.
    """

    def __init__(self, batch_size: int = 200, enable_enhancements: bool = True):
        self.batch_size = batch_size  # Apple handles large batches efficiently
        self.enable_enhancements = enable_enhancements
        self.preprocessing_stats = PreprocessingStats()
        self._binary_path = _ensure_binary()
        # No model/tokenizer to load — verify binary works
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

        # Merge subtitle fragments into sentence-level units
        merged_texts, groups = merge_for_translation(texts)
        logger.debug(
            f'Sentence merging: {len(texts)} lines → {len(merged_texts)} translation units'
        )

        # Apply enhancements: placeholders + preprocessing
        placeholder_mappings: list[dict[str, str]] = []
        skip_indices: set[int] = set()
        cached_translations: dict[int, str] = {}

        if self.enable_enhancements:
            processed_texts = []
            for i, text in enumerate(merged_texts):
                # Extract placeholders (numbers, URLs, names)
                protected, mapping = extract_placeholders(text, self.preprocessing_stats)
                placeholder_mappings.append(mapping)

                # Check for direct phrase mappings
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

        # Batch translate via Swift binary
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

            # Map results back to original indices
            for j, translated in enumerate(batch_results):
                original_idx = texts_to_translate[batch_start + j][0]
                translations[original_idx] = translated

            # Progress callback — count original lines covered
            if progress_callback:
                # Determine how many merged units are done
                units_done = batch_end + len(skip_indices)
                lines_done = min(
                    sum(len(g.line_indices) for g in groups[:units_done]),
                    total_lines,
                )
                elapsed = time.time() - start_time
                rate = lines_done / elapsed if elapsed > 0 else 0
                progress_callback(lines_done, total_lines, rate)

        # Apply postprocessing
        if self.enable_enhancements:
            translations = [postprocess_translation(t) for t in translations]

        # Restore placeholders
        translations = [
            restore_placeholders(text, mapping)
            for text, mapping in zip(translations, placeholder_mappings, strict=True)
        ]

        # Apply fallbacks (empty/suspicious translations → original text)
        translations = _apply_fallbacks(merged_texts, translations, skip_indices, cached_translations)

        # Log stats
        if self.enable_enhancements and self.preprocessing_stats.total_processed > 0:
            logger.info(self.preprocessing_stats.get_summary())

        # Unmerge back to original line count
        return unmerge_translations(translations, groups, texts)

    def cleanup(self) -> None:
        """No-op — Apple backend is stateless (no model to unload)."""
        pass


def _apply_fallbacks(
    originals: list[str],
    translations: list[str],
    skip_indices: set[int],
    cached_translations: dict[int, str],
) -> list[str]:
    """Apply fallback logic: empty/suspicious translations → original text."""
    result = []
    for i, (original, translated) in enumerate(zip(originals, translations, strict=True)):
        if i in skip_indices:
            result.append(cached_translations.get(i, translated))
            continue

        stripped = translated.strip()
        if not stripped:
            logger.warning(f'Empty Apple translation for: "{original}" — using original')
            result.append(original)
        elif len(stripped) < 2 and len(original.strip()) > 5:
            logger.warning(
                f'Suspiciously short Apple translation: "{original}" → "{translated}" — using original'
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
        not _SWIFT_BINARY.exists()
        or _SWIFT_SOURCE.stat().st_mtime > _SWIFT_BINARY.stat().st_mtime
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
                '-framework', 'Translation',
                str(_SWIFT_SOURCE),
                '-o', str(_SWIFT_BINARY),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f'Failed to compile Apple Translation bridge:\n{result.stderr}'
            )

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
        )

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


# ─── Caching ──────────────────────────────────────────────────────────────────

_cached_apple_backend: AppleTranslationBackend | None = None


def _get_apple_backend(batch_size: int) -> AppleTranslationBackend | None:
    """Return a cached Apple backend instance."""
    global _cached_apple_backend
    if _cached_apple_backend is not None and _cached_apple_backend.batch_size == batch_size:
        _cached_apple_backend.preprocessing_stats.reset()
        return _cached_apple_backend

    try:
        backend = AppleTranslationBackend(batch_size=batch_size)
        _cached_apple_backend = backend
        return backend
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(f'Apple Translation backend unavailable: {e}')
        return None
```

### Changes to `movie_translator/translation/translator.py`

Only the `translate_dialogue_lines()` function changes. The `SubtitleTranslator` class is untouched:

```python
# Add import at top:
from .apple_backend import _get_apple_backend

# Replace translate_dialogue_lines() (lines 339-356):
def translate_dialogue_lines(
    dialogue_lines: list[DialogueLine],
    device: str,
    batch_size: int,
    model: str,
    progress_callback: ProgressCallback | None = None,
) -> list[DialogueLine]:
    if model == 'apple':
        backend = _get_apple_backend(batch_size)
        if backend is None:
            return []
        texts = [line.text for line in dialogue_lines]
        translated_texts = backend.translate_texts(texts, progress_callback)
    else:
        translator = _get_translator(device, batch_size, model)
        if translator is None:
            return []
        texts = [line.text for line in dialogue_lines]
        translated_texts = translator.translate_texts(texts, progress_callback)

    return [
        DialogueLine(line.start_ms, line.end_ms, text)
        for line, text in zip(dialogue_lines, translated_texts, strict=True)
    ]
```

### Changes to `movie_translator/main.py`

```python
# Line ~45, update choices:
parser.add_argument('--model', choices=['allegro', 'apple'], default='allegro')
```

### `.gitignore` addition

```
# Compiled Swift binary (auto-built from source)
movie_translator/translation/swift/translate_bridge
```

---

## Performance Comparison

| Metric | Allegro (HuggingFace) | Apple Translation |
|--------|-----------------------|-------------------|
| Speed (300 lines) | ~45s (MPS), ~120s (CPU) | ~7s |
| Subprocess overhead | N/A | ~225ms per batch |
| Model download | ~300MB (first run) | System-managed (~200MB) |
| GPU/MPS required | Yes (for reasonable speed) | No |
| Quality | Good for anime dialogue | Good for general text |
| Sentence context | Handles anime idioms well | Generic ML translation |
| Cost | Free (local model) | Free (on-device) |
| Platform | macOS, Linux | macOS 26+ only |
| Dependencies | torch, transformers, sentencepiece | Swift compiler (Xcode CLI tools) |

---

## Test Plan

### `movie_translator/translation/tests/test_apple_backend.py`

```python
"""Tests for AppleTranslationBackend.

Platform-dependent tests (require macOS 26+ with languages installed)
are marked with @pytest.mark.apple_translation and skipped on CI.
"""

import json
import platform
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from movie_translator.translation.apple_backend import (
    AppleTranslationBackend,
    AppleTranslationError,
    _apply_fallbacks,
    _call_swift_binary,
    _ensure_binary,
    is_available,
)
from movie_translator.types import DialogueLine

# Skip all integration tests if not on macOS 26+
apple_translation = pytest.mark.skipif(
    not is_available(),
    reason='Requires macOS 26+ with Apple Translation',
)


class TestIsAvailable:
    def test_returns_false_on_linux(self):
        with patch('platform.system', return_value='Linux'):
            assert is_available() is False

    def test_returns_false_on_old_macos(self):
        with patch('platform.system', return_value='Darwin'):
            with patch('platform.mac_ver', return_value=('15.4', ('', '', ''), '')):
                assert is_available() is False

    def test_returns_true_on_macos_26(self):
        with patch('platform.system', return_value='Darwin'):
            with patch('platform.mac_ver', return_value=('26.4', ('', '', ''), '')):
                # Also need swift source to exist
                with patch.object(Path, 'exists', return_value=True):
                    assert is_available() is True


class TestCallSwiftBinary:
    def test_parses_success_response(self):
        mock_result = MagicMock()
        mock_result.stdout = json.dumps({
            'translations': ['Cześć', 'Do widzenia'],
            'elapsed_ms': 100,
        }).encode()
        mock_result.returncode = 0

        with patch('subprocess.run', return_value=mock_result):
            result = _call_swift_binary(Path('/fake/binary'), ['Hello', 'Goodbye'])

        assert result == ['Cześć', 'Do widzenia']

    def test_raises_on_error_response(self):
        mock_result = MagicMock()
        mock_result.stdout = json.dumps({
            'error': 'Languages not installed',
            'code': 'not_installed',
        }).encode()
        mock_result.returncode = 1

        with patch('subprocess.run', return_value=mock_result):
            with pytest.raises(AppleTranslationError, match='not installed'):
                _call_swift_binary(Path('/fake/binary'), ['Hello'])

    def test_raises_on_empty_output(self):
        mock_result = MagicMock()
        mock_result.stdout = b''
        mock_result.stderr = b'crash'
        mock_result.returncode = 1

        with patch('subprocess.run', return_value=mock_result):
            with pytest.raises(AppleTranslationError, match='no output'):
                _call_swift_binary(Path('/fake/binary'), ['Hello'])

    def test_raises_on_count_mismatch(self):
        mock_result = MagicMock()
        mock_result.stdout = json.dumps({
            'translations': ['Cześć'],  # sent 2, got 1
            'elapsed_ms': 50,
        }).encode()
        mock_result.returncode = 0

        with patch('subprocess.run', return_value=mock_result):
            with pytest.raises(AppleTranslationError, match='count mismatch'):
                _call_swift_binary(Path('/fake/binary'), ['Hello', 'Goodbye'])


class TestApplyFallbacks:
    def test_empty_translation_uses_original(self):
        result = _apply_fallbacks(
            originals=['Hello world'],
            translations=[''],
            skip_indices=set(),
            cached_translations={},
        )
        assert result == ['Hello world']

    def test_suspiciously_short_uses_original(self):
        result = _apply_fallbacks(
            originals=['What a beautiful day'],
            translations=['X'],
            skip_indices=set(),
            cached_translations={},
        )
        assert result == ['What a beautiful day']

    def test_valid_short_translation_kept(self):
        result = _apply_fallbacks(
            originals=['Yes'],
            translations=['Tak'],
            skip_indices=set(),
            cached_translations={},
        )
        assert result == ['Tak']

    def test_cached_translation_used(self):
        result = _apply_fallbacks(
            originals=['Thank you'],
            translations=['ignored'],
            skip_indices={0},
            cached_translations={0: 'Dziękuję'},
        )
        assert result == ['Dziękuję']


class TestEnsureBinary:
    def test_compiles_when_missing(self, tmp_path):
        source = tmp_path / 'translate_bridge.swift'
        source.write_text('// test')
        binary = tmp_path / 'translate_bridge'

        with (
            patch('movie_translator.translation.apple_backend._SWIFT_SOURCE', source),
            patch('movie_translator.translation.apple_backend._SWIFT_BINARY', binary),
            patch('shutil.which', return_value='/usr/bin/swiftc'),
            patch('subprocess.run') as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stderr='')
            # Touch binary so the exists() check passes after "compilation"
            binary.touch()
            _ensure_binary()
            mock_run.assert_called_once()
            assert '-framework' in mock_run.call_args[0][0]

    def test_skips_compile_when_binary_newer(self, tmp_path):
        source = tmp_path / 'translate_bridge.swift'
        source.write_text('// test')
        binary = tmp_path / 'translate_bridge'
        binary.touch()
        # Make binary newer
        import os
        os.utime(binary, (source.stat().st_mtime + 10, source.stat().st_mtime + 10))

        with (
            patch('movie_translator.translation.apple_backend._SWIFT_SOURCE', source),
            patch('movie_translator.translation.apple_backend._SWIFT_BINARY', binary),
            patch('subprocess.run') as mock_run,
        ):
            _ensure_binary()
            mock_run.assert_not_called()


class TestAppleTranslationBackend:
    def test_translate_texts_with_mocked_binary(self):
        mock_result = MagicMock()
        mock_result.stdout = json.dumps({
            'translations': ['Cześć, jak się masz?', 'Dobrze, dziękuję.'],
            'elapsed_ms': 100,
        }).encode()
        mock_result.returncode = 0

        with (
            patch('movie_translator.translation.apple_backend._ensure_binary', return_value=Path('/fake')),
            patch('subprocess.run', return_value=mock_result),
        ):
            backend = AppleTranslationBackend(batch_size=200, enable_enhancements=False)
            result = backend.translate_texts(['Hello, how are you?', 'Fine, thank you.'])

        assert len(result) == 2
        assert 'Cześć' in result[0]

    def test_empty_input(self):
        with patch('movie_translator.translation.apple_backend._ensure_binary', return_value=Path('/fake')):
            backend = AppleTranslationBackend(enable_enhancements=False)
            assert backend.translate_texts([]) == []

    def test_progress_callback_called(self):
        mock_result = MagicMock()
        mock_result.stdout = json.dumps({
            'translations': ['Cześć'],
            'elapsed_ms': 50,
        }).encode()
        mock_result.returncode = 0

        callback = MagicMock()

        with (
            patch('movie_translator.translation.apple_backend._ensure_binary', return_value=Path('/fake')),
            patch('subprocess.run', return_value=mock_result),
        ):
            backend = AppleTranslationBackend(enable_enhancements=False)
            backend.translate_texts(['Hello'], progress_callback=callback)

        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0] > 0  # lines_done
        assert args[1] == 1  # total_lines


@apple_translation
class TestAppleTranslationIntegration:
    """Integration tests — only run on macOS 26+ with languages installed."""

    def test_single_translation(self):
        backend = AppleTranslationBackend(enable_enhancements=False)
        result = backend.translate_texts(['Hello world'])
        assert len(result) == 1
        assert result[0]  # non-empty
        assert result[0] != 'Hello world'  # actually translated

    def test_batch_translation(self):
        backend = AppleTranslationBackend(enable_enhancements=False)
        texts = ['What on earth?!', 'Look over there!', 'Guess I will go.']
        result = backend.translate_texts(texts)
        assert len(result) == 3
        assert all(t for t in result)  # all non-empty

    def test_with_enhancements(self):
        backend = AppleTranslationBackend(enable_enhancements=True)
        result = backend.translate_texts(['Yes', 'Thank you', 'What a beautiful day!'])
        assert len(result) == 3
```

---

## Action Plan

### Phase 1: Swift Bridge Binary

1. Create directory `movie_translator/translation/swift/`
2. Write `translate_bridge.swift` — the complete source from the "Swift CLI Binary" section above
3. Add `translate_bridge` (compiled binary) to `.gitignore`
4. Verify: `echo '{"texts":["Hello"],"source":"en","target":"pl"}' | swiftc -parse-as-library -O -framework Translation translate_bridge.swift -o /tmp/tb && echo '{"texts":["Hello"],"source":"en","target":"pl"}' | /tmp/tb`

### Phase 2: Apple Backend Module

1. Create `movie_translator/translation/apple_backend.py` — the complete module from the "New Files" section above
2. This includes: `AppleTranslationBackend`, `is_available()`, `check_languages_installed()`, `_ensure_binary()`, `_call_swift_binary()`, `_apply_fallbacks()`, and caching via `_get_apple_backend()`

### Phase 3: Wire Into Existing Code

1. **`movie_translator/translation/translator.py`** — update `translate_dialogue_lines()` to dispatch on `model == 'apple'`. Add `from .apple_backend import _get_apple_backend` import.
2. **`movie_translator/main.py`** — add `'apple'` to argparse `choices` for `--model`
3. **`.gitignore`** — add `movie_translator/translation/swift/translate_bridge`

### Phase 4: Tests

1. Create `movie_translator/translation/tests/test_apple_backend.py` — the complete test file from the "Test Plan" section above
2. Run: `pytest movie_translator/translation/tests/test_apple_backend.py -v`
3. On macOS 26+ with languages: `pytest movie_translator/translation/tests/test_apple_backend.py -v -m apple_translation`

### Phase 5: End-to-End Verification

1. Test with an actual video file: `python -m movie_translator --model apple <video.mkv>`
2. Compare output quality against Allegro: run same file with `--model allegro`
3. Verify progress display works in the TUI

---

## Open Questions

1. **Should Apple be the default on macOS 26+?** It's 6x faster but may have lower quality for anime-specific dialogue. Could A/B test on a few episodes and compare.

2. **Persistent process vs one-shot?** Current design spawns a new subprocess per batch. Could keep a long-running Swift process with line-delimited JSON for lower latency. Probably unnecessary given ~225ms startup overhead vs ~7s translation time per episode.

3. **Enhancement layer compatibility?** The idiom patterns and phrase maps in `enhancements.py` were tuned for the Allegro model. Apple Translation may handle some idioms better natively, or may need its own tuning. Test and decide.

4. **Binary distribution?** Recommend compile from source on first run. The `_ensure_binary()` function handles this automatically. The source file is tiny (~50 lines) and compiles in <2 seconds.
