# Subtitle Validation via Timing Fingerprinting — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate downloaded subtitles against embedded/OCR reference tracks using timing pattern fingerprinting, rejecting wrong-episode matches and selecting the best candidate from all plausible downloads.

**Architecture:** New `SubtitleValidator` module computes binary speech-activity vectors from subtitle timing, then uses normalized cross-correlation to score candidates against a reference track. The fetcher is refactored to expose all candidates (not just the best), and the pipeline orchestrates a download-all → validate-all → select-best flow. Per-video working directories manage intermediate artifacts.

**Tech Stack:** Python 3.10+, pysubs2 (subtitle parsing), numpy (cross-correlation), pytest (testing)

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `movie_translator/subtitle_fetch/validator.py` | Fingerprinting algorithm, cross-correlation scoring, candidate ranking |
| Create | `movie_translator/subtitle_fetch/tests/__init__.py` | Test package init |
| Create | `movie_translator/subtitle_fetch/tests/test_validator.py` | Unit tests for validator |
| Create | `movie_translator/tests/test_working_dir.py` | Unit tests for working directory management |
| Modify | `movie_translator/subtitle_fetch/fetcher.py` | Add `search_all()` and `download_candidate()` methods |
| Modify | `movie_translator/subtitle_fetch/__init__.py` | Export `SubtitleValidator` |
| Modify | `movie_translator/pipeline.py` | New validation flow, working directory management, reference extraction moved earlier |
| Modify | `movie_translator/main.py` | `--keep-artifacts` flag, per-anime/per-episode working directory structure |

---

### Task 1: Timing Fingerprint — Build Activity Vector

**Files:**
- Create: `movie_translator/subtitle_fetch/tests/__init__.py`
- Create: `movie_translator/subtitle_fetch/tests/test_validator.py`
- Create: `movie_translator/subtitle_fetch/validator.py`

- [ ] **Step 1: Write failing test for activity vector generation**

```python
# movie_translator/subtitle_fetch/tests/test_validator.py
import numpy as np

from movie_translator.subtitle_fetch.validator import build_activity_vector


def test_build_activity_vector_basic():
    """A single dialogue line from 1000ms-3000ms with 2s bins should activate bins 0 and 1."""
    timestamps = [(1000, 3000)]  # one line: 1s to 3s
    duration_ms = 6000
    bin_size_ms = 2000
    vec = build_activity_vector(timestamps, duration_ms, bin_size_ms)
    # Bins: [0-2000ms], [2000-4000ms], [4000-6000ms]
    # Line 1000-3000 overlaps bin 0 and bin 1
    assert vec.shape == (3,)
    assert vec[0] == 1  # 1000-3000 overlaps 0-2000
    assert vec[1] == 1  # 1000-3000 overlaps 2000-4000
    assert vec[2] == 0  # nothing in 4000-6000


def test_build_activity_vector_empty():
    """No timestamps should produce all-zero vector."""
    vec = build_activity_vector([], 6000, 2000)
    assert vec.shape == (3,)
    assert np.all(vec == 0)


def test_build_activity_vector_multiple_lines():
    """Multiple dialogue lines scattered across the timeline."""
    timestamps = [(0, 1000), (4000, 5500), (5500, 6000)]
    duration_ms = 8000
    bin_size_ms = 2000
    vec = build_activity_vector(timestamps, duration_ms, bin_size_ms)
    # Bins: [0-2000], [2000-4000], [4000-6000], [6000-8000]
    assert vec.shape == (4,)
    assert vec[0] == 1  # 0-1000 overlaps 0-2000
    assert vec[1] == 0  # nothing in 2000-4000
    assert vec[2] == 1  # 4000-5500 and 5500-6000 overlap 4000-6000
    assert vec[3] == 0  # nothing in 6000-8000
```

Also create the empty test init:
```python
# movie_translator/subtitle_fetch/tests/__init__.py
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/w/h_dev/movie_translator && python -m pytest movie_translator/subtitle_fetch/tests/test_validator.py -v`
Expected: FAIL with `ImportError` — `build_activity_vector` doesn't exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
# movie_translator/subtitle_fetch/validator.py
"""Subtitle validation via timing pattern fingerprinting.

Compares the temporal pattern of dialogue activity between a reference
subtitle track and downloaded candidates using cross-correlation of
binary activity vectors.
"""

import math

import numpy as np


def build_activity_vector(
    timestamps: list[tuple[int, int]],
    duration_ms: int,
    bin_size_ms: int = 2000,
) -> np.ndarray:
    """Convert subtitle timestamps to a binary activity vector.

    Divides the timeline into fixed-width bins. Each bin is 1 if any
    dialogue overlaps it, 0 otherwise.

    Args:
        timestamps: List of (start_ms, end_ms) tuples.
        duration_ms: Total timeline duration in milliseconds.
        bin_size_ms: Width of each time bin in milliseconds.

    Returns:
        Binary numpy array of shape (n_bins,).
    """
    n_bins = max(1, math.ceil(duration_ms / bin_size_ms))
    vec = np.zeros(n_bins, dtype=np.float64)

    for start, end in timestamps:
        first_bin = max(0, start // bin_size_ms)
        last_bin = min(n_bins - 1, (end - 1) // bin_size_ms) if end > start else first_bin
        vec[first_bin : last_bin + 1] = 1

    return vec
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/w/h_dev/movie_translator && python -m pytest movie_translator/subtitle_fetch/tests/test_validator.py -v`
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add movie_translator/subtitle_fetch/validator.py movie_translator/subtitle_fetch/tests/__init__.py movie_translator/subtitle_fetch/tests/test_validator.py
git commit -m "feat(validator): add activity vector generation for timing fingerprinting"
```

---

### Task 2: Cross-Correlation Scoring

**Files:**
- Modify: `movie_translator/subtitle_fetch/tests/test_validator.py`
- Modify: `movie_translator/subtitle_fetch/validator.py`

- [ ] **Step 1: Write failing tests for cross-correlation scoring**

Append to `movie_translator/subtitle_fetch/tests/test_validator.py`:

```python
from movie_translator.subtitle_fetch.validator import compute_similarity


def test_identical_vectors_score_1():
    """Identical activity patterns should score 1.0."""
    vec = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1], dtype=np.float64)
    score = compute_similarity(vec, vec)
    assert score >= 0.99


def test_opposite_vectors_score_low():
    """Completely opposite patterns should score near 0."""
    vec_a = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float64)
    vec_b = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float64)
    score = compute_similarity(vec_a, vec_b)
    assert score < 0.2


def test_shifted_vectors_score_high():
    """A shifted version of the same pattern should still score high (cross-correlation finds best offset)."""
    base = np.array([1, 1, 0, 0, 1, 0, 1, 1, 0, 0], dtype=np.float64)
    shifted = np.zeros(10, dtype=np.float64)
    shifted[1:] = base[:-1]  # shift right by 1 bin
    score = compute_similarity(base, shifted)
    assert score >= 0.7


def test_empty_reference_scores_zero():
    """If reference has no activity, score should be 0."""
    ref = np.zeros(10, dtype=np.float64)
    cand = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1], dtype=np.float64)
    score = compute_similarity(ref, cand)
    assert score == 0.0


def test_empty_candidate_scores_zero():
    """If candidate has no activity, score should be 0."""
    ref = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1], dtype=np.float64)
    cand = np.zeros(10, dtype=np.float64)
    score = compute_similarity(ref, cand)
    assert score == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/w/h_dev/movie_translator && python -m pytest movie_translator/subtitle_fetch/tests/test_validator.py::test_identical_vectors_score_1 -v`
Expected: FAIL with `ImportError` — `compute_similarity` doesn't exist yet.

- [ ] **Step 3: Implement cross-correlation scoring**

Add to `movie_translator/subtitle_fetch/validator.py`:

```python
def compute_similarity(
    reference: np.ndarray,
    candidate: np.ndarray,
    max_shift_bins: int = 15,
) -> float:
    """Compute normalized cross-correlation between two activity vectors.

    Tries shifting the candidate by up to max_shift_bins in either direction
    to find the best alignment. Returns the peak normalized correlation.

    Args:
        reference: Binary activity vector for the reference subtitle.
        candidate: Binary activity vector for the candidate subtitle.
        max_shift_bins: Maximum shift to try in either direction.

    Returns:
        Similarity score from 0.0 (no match) to 1.0 (perfect match).
    """
    ref_energy = np.sum(reference)
    cand_energy = np.sum(candidate)

    if ref_energy == 0 or cand_energy == 0:
        return 0.0

    # Normalize by geometric mean of energies
    norm = np.sqrt(ref_energy * cand_energy)

    best_score = 0.0
    n = len(reference)
    m = len(candidate)

    for shift in range(-max_shift_bins, max_shift_bins + 1):
        # Compute overlap region
        ref_start = max(0, shift)
        ref_end = min(n, m + shift)
        cand_start = max(0, -shift)
        cand_end = cand_start + (ref_end - ref_start)

        if ref_start >= ref_end:
            continue

        overlap = np.sum(reference[ref_start:ref_end] * candidate[cand_start:cand_end])
        score = overlap / norm
        best_score = max(best_score, score)

    return float(min(best_score, 1.0))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/w/h_dev/movie_translator && python -m pytest movie_translator/subtitle_fetch/tests/test_validator.py -v`
Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add movie_translator/subtitle_fetch/validator.py movie_translator/subtitle_fetch/tests/test_validator.py
git commit -m "feat(validator): add cross-correlation similarity scoring"
```

---

### Task 3: Subtitle File Parsing to Timestamps

**Files:**
- Modify: `movie_translator/subtitle_fetch/tests/test_validator.py`
- Modify: `movie_translator/subtitle_fetch/validator.py`

- [ ] **Step 1: Write failing test for subtitle timestamp extraction**

This test needs a real subtitle file. Create a minimal SRT in a tmp directory.

Append to `movie_translator/subtitle_fetch/tests/test_validator.py`:

```python
from pathlib import Path
import tempfile

from movie_translator.subtitle_fetch.validator import extract_timestamps


def test_extract_timestamps_from_srt(tmp_path):
    """Parse an SRT file and extract dialogue timestamps."""
    srt_content = """1
00:00:01,000 --> 00:00:03,000
Hello world

2
00:00:05,000 --> 00:00:07,500
How are you?

3
00:00:10,000 --> 00:00:12,000
Goodbye
"""
    srt_file = tmp_path / "test.srt"
    srt_file.write_text(srt_content)

    timestamps, duration_ms = extract_timestamps(srt_file)
    assert len(timestamps) == 3
    assert timestamps[0] == (1000, 3000)
    assert timestamps[1] == (5000, 7500)
    assert timestamps[2] == (10000, 12000)
    assert duration_ms == 12000


def test_extract_timestamps_filters_non_dialogue(tmp_path):
    """ASS files should have non-dialogue events filtered out."""
    ass_content = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,1,2,10,10,40,1
Style: Sign,Arial,36,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,1,2,10,10,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Hello world
Dialogue: 0,0:00:04.00,0:00:06.00,Sign,,0,0,0,,{\\pos(640,100)}Store Name
Dialogue: 0,0:00:08.00,0:00:10.00,Default,,0,0,0,,Goodbye
"""
    ass_file = tmp_path / "test.ass"
    ass_file.write_text(ass_content)

    timestamps, duration_ms = extract_timestamps(ass_file)
    # Sign event should be filtered out
    assert len(timestamps) == 2
    assert timestamps[0] == (1000, 3000)
    assert timestamps[1] == (8000, 10000)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/w/h_dev/movie_translator && python -m pytest movie_translator/subtitle_fetch/tests/test_validator.py::test_extract_timestamps_from_srt -v`
Expected: FAIL with `ImportError` — `extract_timestamps` doesn't exist yet.

- [ ] **Step 3: Implement timestamp extraction**

Add to `movie_translator/subtitle_fetch/validator.py`:

```python
from pathlib import Path

from ..subtitles._pysubs2 import get_pysubs2
from ..types import NON_DIALOGUE_STYLES


def extract_timestamps(subtitle_path: Path) -> tuple[list[tuple[int, int]], int]:
    """Extract dialogue timestamps from a subtitle file.

    Parses the subtitle file, filters out non-dialogue events (signs, songs, etc.),
    and returns timing pairs.

    Args:
        subtitle_path: Path to subtitle file (SRT, ASS, or any pysubs2-supported format).

    Returns:
        Tuple of (timestamps, duration_ms) where timestamps is a list of (start_ms, end_ms)
        and duration_ms is the end time of the last event.
    """
    pysubs2 = get_pysubs2()
    if pysubs2 is None:
        return [], 0

    subs = pysubs2.load(str(subtitle_path))

    timestamps = []
    for event in subs:
        if not event.text or not event.text.strip():
            continue

        # Filter non-dialogue styles
        style = getattr(event, 'style', 'Default').lower()
        if any(keyword in style for keyword in NON_DIALOGUE_STYLES):
            continue

        # Skip empty plaintext (after stripping ASS tags)
        if hasattr(event, 'plaintext') and not event.plaintext.strip():
            continue

        timestamps.append((event.start, event.end))

    duration_ms = max(end for _, end in timestamps) if timestamps else 0
    return timestamps, duration_ms
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/w/h_dev/movie_translator && python -m pytest movie_translator/subtitle_fetch/tests/test_validator.py -v`
Expected: All 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add movie_translator/subtitle_fetch/validator.py movie_translator/subtitle_fetch/tests/test_validator.py
git commit -m "feat(validator): add subtitle timestamp extraction with dialogue filtering"
```

---

### Task 4: SubtitleValidator Class — Tying It Together

**Files:**
- Modify: `movie_translator/subtitle_fetch/tests/test_validator.py`
- Modify: `movie_translator/subtitle_fetch/validator.py`

- [ ] **Step 1: Write failing test for the SubtitleValidator class**

Append to `movie_translator/subtitle_fetch/tests/test_validator.py`:

```python
from movie_translator.subtitle_fetch.validator import SubtitleValidator


def _make_srt(tmp_path: Path, name: str, lines: list[tuple[str, str, str]]) -> Path:
    """Helper: create an SRT file from (start, end, text) tuples.

    Times are in SRT format like '00:00:01,000'.
    """
    content = ""
    for i, (start, end, text) in enumerate(lines, 1):
        content += f"{i}\n{start} --> {end}\n{text}\n\n"
    path = tmp_path / name
    path.write_text(content)
    return path


def test_validator_scores_matching_subtitle_high(tmp_path):
    """A candidate with similar timing to reference should score high."""
    ref_lines = [
        ("00:00:01,000", "00:00:03,000", "Hello"),
        ("00:00:10,000", "00:00:12,000", "Middle"),
        ("00:00:20,000", "00:00:22,000", "End"),
    ]
    # Same timing pattern, different text (like a translation)
    cand_lines = [
        ("00:00:01,200", "00:00:03,200", "Halo"),
        ("00:00:10,100", "00:00:12,100", "Srodek"),
        ("00:00:20,000", "00:00:22,000", "Koniec"),
    ]

    ref_path = _make_srt(tmp_path, "ref.srt", ref_lines)
    cand_path = _make_srt(tmp_path, "cand.srt", cand_lines)

    validator = SubtitleValidator(ref_path)
    score = validator.score_candidate(cand_path)
    assert score >= 0.8


def test_validator_scores_wrong_episode_low(tmp_path):
    """A candidate with completely different timing should score low."""
    ref_lines = [
        ("00:00:01,000", "00:00:03,000", "Hello"),
        ("00:00:10,000", "00:00:12,000", "Middle"),
        ("00:00:20,000", "00:00:22,000", "End"),
    ]
    # Completely different timing pattern
    wrong_lines = [
        ("00:00:05,000", "00:00:07,000", "Wrong ep line 1"),
        ("00:00:15,000", "00:00:17,000", "Wrong ep line 2"),
        ("00:00:25,000", "00:00:27,000", "Wrong ep line 3"),
    ]

    ref_path = _make_srt(tmp_path, "ref.srt", ref_lines)
    wrong_path = _make_srt(tmp_path, "wrong.srt", wrong_lines)

    validator = SubtitleValidator(ref_path)
    score = validator.score_candidate(wrong_path)
    assert score < 0.5


def test_validator_validate_candidates_ranking(tmp_path):
    """validate_candidates should return candidates sorted by score, filtered by threshold."""
    ref_lines = [
        ("00:00:01,000", "00:00:03,000", "Hello"),
        ("00:00:10,000", "00:00:12,000", "Middle"),
        ("00:00:20,000", "00:00:22,000", "End"),
    ]
    good_lines = [
        ("00:00:01,200", "00:00:03,200", "Halo"),
        ("00:00:10,100", "00:00:12,100", "Srodek"),
        ("00:00:20,000", "00:00:22,000", "Koniec"),
    ]
    bad_lines = [
        ("00:00:05,000", "00:00:07,000", "Zly"),
        ("00:00:15,000", "00:00:17,000", "Odcinek"),
    ]

    ref_path = _make_srt(tmp_path, "ref.srt", ref_lines)
    good_path = _make_srt(tmp_path, "good.srt", good_lines)
    bad_path = _make_srt(tmp_path, "bad.srt", bad_lines)

    from movie_translator.subtitle_fetch.types import SubtitleMatch

    good_match = SubtitleMatch(
        language="pol", source="test", subtitle_id="1",
        release_name="good", format="srt", score=0.6, hash_match=False,
    )
    bad_match = SubtitleMatch(
        language="pol", source="test", subtitle_id="2",
        release_name="bad", format="srt", score=0.7, hash_match=False,
    )

    validator = SubtitleValidator(ref_path)
    results = validator.validate_candidates(
        [(bad_match, bad_path), (good_match, good_path)],
        min_threshold=0.3,
    )

    # Good candidate should be first (higher validation score)
    assert len(results) >= 1
    assert results[0][0].release_name == "good"
    assert results[0][2] > results[-1][2] if len(results) > 1 else True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/w/h_dev/movie_translator && python -m pytest movie_translator/subtitle_fetch/tests/test_validator.py::test_validator_scores_matching_subtitle_high -v`
Expected: FAIL with `ImportError` — `SubtitleValidator` doesn't exist yet.

- [ ] **Step 3: Implement SubtitleValidator class**

Add to `movie_translator/subtitle_fetch/validator.py`:

```python
from ..logging import logger
from .types import SubtitleMatch


class SubtitleValidator:
    """Validates downloaded subtitles against a reference track using timing fingerprints."""

    def __init__(self, reference_path: Path, bin_size_ms: int = 2000):
        self._bin_size_ms = bin_size_ms
        self._ref_timestamps, self._ref_duration = extract_timestamps(reference_path)
        self._ref_vector = build_activity_vector(
            self._ref_timestamps, self._ref_duration, bin_size_ms
        )
        logger.debug(
            f'Validator: reference has {len(self._ref_timestamps)} dialogue events, '
            f'{self._ref_duration}ms duration, {len(self._ref_vector)} bins'
        )

    def score_candidate(self, candidate_path: Path) -> float:
        """Score a candidate subtitle against the reference.

        Returns similarity from 0.0 (no match) to 1.0 (perfect match).
        """
        cand_timestamps, cand_duration = extract_timestamps(candidate_path)

        if not cand_timestamps:
            logger.debug(f'Validator: candidate {candidate_path.name} has no dialogue events')
            return 0.0

        # Use the longer duration to ensure vectors cover the same timeline
        duration = max(self._ref_duration, cand_duration)
        ref_vec = build_activity_vector(self._ref_timestamps, duration, self._bin_size_ms)
        cand_vec = build_activity_vector(cand_timestamps, duration, self._bin_size_ms)

        return compute_similarity(ref_vec, cand_vec)

    def validate_candidates(
        self,
        candidates: list[tuple[SubtitleMatch, Path]],
        min_threshold: float = 0.5,
    ) -> list[tuple[SubtitleMatch, Path, float]]:
        """Score all candidates and return those above threshold, sorted by score descending.

        Args:
            candidates: List of (SubtitleMatch, downloaded_file_path) tuples.
            min_threshold: Minimum validation score to include in results.

        Returns:
            List of (SubtitleMatch, Path, validation_score) sorted by score descending.
        """
        scored = []
        for match, path in candidates:
            score = self.score_candidate(path)
            logger.info(
                f'Validation: {match.release_name} ({match.source}/{match.language}) '
                f'→ score={score:.3f}'
            )
            if score >= min_threshold:
                scored.append((match, path, score))
            else:
                logger.debug(
                    f'Validation: rejected {match.release_name} (score {score:.3f} < {min_threshold})'
                )

        scored.sort(key=lambda x: x[2], reverse=True)
        return scored
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/w/h_dev/movie_translator && python -m pytest movie_translator/subtitle_fetch/tests/test_validator.py -v`
Expected: All 13 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add movie_translator/subtitle_fetch/validator.py movie_translator/subtitle_fetch/tests/test_validator.py
git commit -m "feat(validator): add SubtitleValidator class with candidate ranking"
```

---

### Task 5: Refactor Fetcher — Expose All Candidates

**Files:**
- Modify: `movie_translator/subtitle_fetch/fetcher.py`
- Modify: `movie_translator/subtitle_fetch/__init__.py`

- [ ] **Step 1: Write failing test for search_all and download_candidate**

Append to `movie_translator/subtitle_fetch/tests/test_validator.py`:

```python
from unittest.mock import MagicMock
from movie_translator.subtitle_fetch.fetcher import SubtitleFetcher


def test_search_all_returns_all_matches():
    """search_all should return all matches from all providers, not just the best per language."""
    match1 = SubtitleMatch("eng", "prov_a", "1", "Release A", "srt", 0.7, False)
    match2 = SubtitleMatch("eng", "prov_a", "2", "Release B", "srt", 0.6, False)
    match3 = SubtitleMatch("pol", "prov_b", "3", "Release C", "ass", 0.5, False)

    provider_a = MagicMock()
    provider_a.name = "prov_a"
    provider_a.search.return_value = [match1, match2]

    provider_b = MagicMock()
    provider_b.name = "prov_b"
    provider_b.search.return_value = [match3]

    fetcher = SubtitleFetcher([provider_a, provider_b])
    identity = MagicMock()
    results = fetcher.search_all(identity, ["eng", "pol"])

    assert len(results) == 3
    assert match1 in results
    assert match2 in results
    assert match3 in results


def test_download_candidate(tmp_path):
    """download_candidate should delegate to the correct provider."""
    match = SubtitleMatch("eng", "prov_a", "1", "Release A", "srt", 0.7, False)
    output_path = tmp_path / "candidate.srt"

    provider = MagicMock()
    provider.name = "prov_a"
    provider.download.return_value = output_path

    fetcher = SubtitleFetcher([provider])
    result = fetcher.download_candidate(match, output_path)

    provider.download.assert_called_once_with(match, output_path)
    assert result == output_path
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/w/h_dev/movie_translator && python -m pytest movie_translator/subtitle_fetch/tests/test_validator.py::test_search_all_returns_all_matches -v`
Expected: FAIL — `SubtitleFetcher` has no `search_all` method.

- [ ] **Step 3: Add search_all and download_candidate to SubtitleFetcher**

Replace the contents of `movie_translator/subtitle_fetch/fetcher.py`:

```python
from pathlib import Path

from ..logging import logger
from .types import SubtitleMatch


class SubtitleFetcher:
    """Orchestrates subtitle search across multiple providers."""

    def __init__(self, providers: list):
        self._providers = providers

    def search_all(
        self,
        identity,
        languages: list[str],
    ) -> list[SubtitleMatch]:
        """Search all providers and return all plausible matches.

        Returns all matches from all providers, sorted by score descending.
        """
        all_matches: list[SubtitleMatch] = []
        for provider in self._providers:
            try:
                matches = provider.search(identity, languages)
                all_matches.extend(matches)
                logger.debug(f'{provider.name}: found {len(matches)} matches')
            except Exception as e:
                logger.warning(f'{provider.name} search failed: {e}')

        if not all_matches:
            logger.info('No subtitles found from any provider')

        all_matches.sort(key=lambda m: (m.score, m.hash_match), reverse=True)
        return all_matches

    def download_candidate(
        self,
        match: SubtitleMatch,
        output_path: Path,
    ) -> Path:
        """Download a single candidate subtitle file.

        Returns the path written to.
        """
        provider = self._find_provider(match.source)
        if not provider:
            raise RuntimeError(f'No provider found for source: {match.source}')
        return provider.download(match, output_path)

    def fetch_subtitles(
        self,
        identity,
        languages: list[str],
        output_dir: Path,
    ) -> dict[str, Path]:
        """Search all providers and download best subtitle per language.

        Returns {language_code: subtitle_file_path} for successfully downloaded subtitles.
        Legacy method — prefer search_all + download_candidate for validated flows.
        """
        all_matches = self.search_all(identity, languages)
        if not all_matches:
            return {}

        # Pick best match per language (highest score wins)
        best: dict[str, SubtitleMatch] = {}
        for match in all_matches:
            if match.language not in best:
                best[match.language] = match

        # Download best matches
        result: dict[str, Path] = {}
        for lang, match in best.items():
            output_path = output_dir / f'fetched_{lang}.{match.format}'
            try:
                self.download_candidate(match, output_path)
                result[lang] = output_path
                logger.info(
                    f'Fetched {lang} subtitles: {match.release_name} '
                    f'({"hash" if match.hash_match else "query"} match, {match.source})'
                )
            except Exception as e:
                logger.warning(f'Failed to download {lang} subtitle: {e}')

        return result

    def _find_provider(self, name: str):
        for p in self._providers:
            if p.name == name:
                return p
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/w/h_dev/movie_translator && python -m pytest movie_translator/subtitle_fetch/tests/test_validator.py -v`
Expected: All 15 tests PASS.

- [ ] **Step 5: Update __init__.py exports**

Edit `movie_translator/subtitle_fetch/__init__.py`:

```python
from .fetcher import SubtitleFetcher
from .providers.base import SubtitleProvider
from .types import SubtitleMatch
from .validator import SubtitleValidator

__all__ = ['SubtitleFetcher', 'SubtitleMatch', 'SubtitleProvider', 'SubtitleValidator']
```

- [ ] **Step 6: Run all existing tests to check nothing is broken**

Run: `cd /Users/w/h_dev/movie_translator && python -m pytest movie_translator/ -v --ignore=.venv`
Expected: All tests PASS (including pre-existing tests).

- [ ] **Step 7: Commit**

```bash
git add movie_translator/subtitle_fetch/fetcher.py movie_translator/subtitle_fetch/__init__.py movie_translator/subtitle_fetch/tests/test_validator.py
git commit -m "refactor(fetcher): add search_all and download_candidate, keep fetch_subtitles as legacy"
```

---

### Task 6: Working Directory Management

**Files:**
- Modify: `movie_translator/main.py`
- Create: `movie_translator/tests/test_working_dir.py`

- [ ] **Step 1: Write failing test for per-anime/per-episode working directory creation**

```python
# movie_translator/tests/test_working_dir.py
from pathlib import Path

from movie_translator.main import create_working_dirs


def test_creates_per_anime_per_episode_dirs(tmp_path):
    """Working dirs should nest as .translate_temp/<anime>/<episode>/."""
    anime_dir = tmp_path / "Aho-Girl"
    anime_dir.mkdir()
    video = anime_dir / "Aho-Girl - 01.mkv"
    video.touch()

    work_dir = create_working_dirs(video, tmp_path)

    assert work_dir.parent.name == "Aho-Girl"  # per-anime parent
    assert work_dir.name == "Aho-Girl - 01"  # per-episode dir
    assert work_dir.parent.parent.name == ".translate_temp"
    assert (work_dir / "candidates").is_dir()
    assert (work_dir / "reference").is_dir()


def test_creates_subdirs(tmp_path):
    """Working dir should have candidates/ and reference/ subdirs."""
    anime_dir = tmp_path / "Show"
    anime_dir.mkdir()
    video = anime_dir / "Show - 01.mkv"
    video.touch()

    work_dir = create_working_dirs(video, tmp_path)

    assert (work_dir / "candidates").is_dir()
    assert (work_dir / "reference").is_dir()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/w/h_dev/movie_translator && python -m pytest movie_translator/tests/test_working_dir.py -v`
Expected: FAIL — `create_working_dirs` doesn't exist.

- [ ] **Step 3: Implement create_working_dirs in main.py**

Add to `movie_translator/main.py` (before the `find_video_files_with_temp_dirs` function):

```python
def create_working_dirs(video_path: Path, input_dir: Path) -> Path:
    """Create per-anime/per-episode working directory structure.

    Structure: input_dir/.translate_temp/<anime_folder>/<video_stem>/
    With subdirs: candidates/, reference/

    Args:
        video_path: Path to the video file.
        input_dir: Root input directory.

    Returns:
        Path to the per-episode working directory.
    """
    temp_root = input_dir / '.translate_temp'
    anime_name = video_path.parent.name
    episode_name = video_path.stem

    work_dir = temp_root / anime_name / episode_name
    (work_dir / 'candidates').mkdir(parents=True, exist_ok=True)
    (work_dir / 'reference').mkdir(parents=True, exist_ok=True)

    return work_dir
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/w/h_dev/movie_translator && python -m pytest movie_translator/tests/test_working_dir.py -v`
Expected: 2 tests PASS.

- [ ] **Step 5: Add --keep-artifacts CLI flag**

In `movie_translator/main.py`, add to `parse_args()` after the `--dry-run` argument:

```python
    parser.add_argument(
        '--keep-artifacts',
        action='store_true',
        help='Preserve working directories after processing (default: clean up on success)',
    )
```

- [ ] **Step 6: Commit**

```bash
git add movie_translator/main.py movie_translator/tests/test_working_dir.py
git commit -m "feat(main): add per-anime/per-episode working directories and --keep-artifacts flag"
```

---

### Task 7: Pipeline Integration — Validated Subtitle Flow

**Files:**
- Modify: `movie_translator/pipeline.py`

This is the largest change — rewiring the pipeline to use the new validation flow.

- [ ] **Step 1: Refactor pipeline to extract reference track first**

Edit `movie_translator/pipeline.py`. The new `process_video_file` method needs a `work_dir` parameter (the per-episode working directory) instead of `temp_dir`. Here is the complete replacement for the class:

```python
import os
import shutil
from pathlib import Path

from .fonts import check_embedded_fonts_support_polish
from .identifier import identify_media
from .inpainting import remove_burned_in_subtitles
from .logging import logger
from .ocr import extract_burned_in_subtitles, is_vision_ocr_available, probe_for_burned_in_subtitles
from .subtitle_fetch import SubtitleFetcher, SubtitleValidator
from .subtitle_fetch.providers.animesub import AnimeSubProvider
from .subtitle_fetch.providers.opensubtitles import OpenSubtitlesProvider
from .subtitles import SubtitleExtractor, SubtitleProcessor
from .translation import translate_dialogue_lines
from .types import OCRResult
from .video import VideoOperations


class TranslationPipeline:
    def __init__(
        self,
        device: str = 'mps',
        batch_size: int = 16,
        model: str = 'allegro',
        enable_fetch: bool = True,
        tracker=None,
    ):
        self.device = device
        self.batch_size = batch_size
        self.model = model
        self.enable_fetch = enable_fetch
        self.tracker = tracker
        self._extractor = None
        self._video_ops = None
        self._ocr_results: list[OCRResult] | None = None

    def _stage(self, name: str, info: str = ''):
        if self.tracker:
            self.tracker.set_stage(name, info)

    def _build_fetcher(self) -> SubtitleFetcher | None:
        if not self.enable_fetch:
            return None

        providers: list = [AnimeSubProvider()]
        api_key = os.environ.get('OPENSUBTITLES_API_KEY', '')
        if api_key:
            providers.append(OpenSubtitlesProvider(api_key=api_key))

        return SubtitleFetcher(providers)

    def _extract_reference(self, video_path: Path, work_dir: Path) -> Path | None:
        """Extract reference subtitle track for validation.

        Tries embedded text subtitle first, then OCR. Returns path to reference
        subtitle file, or None if no reference is available.
        """
        extractor = self._get_extractor()
        ref_dir = work_dir / 'reference'

        track_info = extractor.get_track_info(video_path)
        if track_info:
            eng_track = extractor.find_english_track(track_info)
            if eng_track:
                track_id = eng_track['id']
                subtitle_ext = extractor.get_subtitle_extension(eng_track)
                ref_path = ref_dir / f'embedded_eng{subtitle_ext}'
                subtitle_index = eng_track.get('subtitle_index', 0)
                extractor.extract_subtitle(video_path, track_id, ref_path, subtitle_index)
                logger.info(f'Reference: embedded English track ({eng_track.get("codec_name", "?")})')
                return ref_path

        # Try OCR as reference
        if is_vision_ocr_available() and probe_for_burned_in_subtitles(video_path):
            logger.info('Reference: attempting OCR extraction...')
            result = extract_burned_in_subtitles(video_path, ref_dir)
            if result is not None:
                self._ocr_results = result.ocr_results
                logger.info('Reference: OCR-extracted subtitles')
                return result.srt_path

        logger.warning('No reference subtitle available — validation will be skipped')
        return None

    def _search_and_validate(
        self,
        video_path: Path,
        work_dir: Path,
        reference_path: Path | None,
    ) -> dict[str, Path]:
        """Search providers, download all candidates, validate against reference.

        Returns {language: best_validated_path} for each language with a valid match.
        """
        fetcher = self._build_fetcher()
        if fetcher is None:
            return {}

        try:
            identity = identify_media(video_path)
        except Exception as e:
            logger.warning(f'Media identification failed: {e}')
            return {}

        # Search all providers
        self._stage('fetch', 'searching')
        all_matches = fetcher.search_all(identity, ['eng', 'pol'])
        if not all_matches:
            return {}

        logger.info(f'Found {len(all_matches)} subtitle candidates')

        # Download all candidates
        self._stage('fetch', 'downloading candidates')
        candidates_dir = work_dir / 'candidates'
        downloaded: list[tuple] = []  # (SubtitleMatch, Path)

        for i, match in enumerate(all_matches):
            filename = f'{match.source}_{match.language}_{i}.{match.format}'
            output_path = candidates_dir / filename
            try:
                fetcher.download_candidate(match, output_path)
                downloaded.append((match, output_path))
            except Exception as e:
                logger.warning(f'Failed to download candidate {match.release_name}: {e}')

        if not downloaded:
            logger.warning('All candidate downloads failed')
            return {}

        # Validate against reference (if available)
        if reference_path is not None:
            self._stage('fetch', 'validating')
            validator = SubtitleValidator(reference_path)
            validated = validator.validate_candidates(downloaded)

            if not validated:
                logger.error('No candidates passed validation — skipping')
                return {}

            # Select best per language
            result: dict[str, Path] = {}
            for match, path, score in validated:
                if match.language not in result:
                    result[match.language] = path
                    logger.info(
                        f'Selected {match.language}: {match.release_name} '
                        f'(score={score:.3f}, {match.source})'
                    )
            return result
        else:
            # No reference — fall back to provider scoring (legacy behavior)
            logger.warning('No reference track — using provider scores only (unvalidated)')
            result: dict[str, Path] = {}
            for match, path in downloaded:
                if match.language not in result:
                    result[match.language] = path
                    logger.info(
                        f'Selected {match.language} (unvalidated): {match.release_name} '
                        f'({match.source})'
                    )
            return result

    def process_video_file(self, video_path: Path, work_dir: Path, dry_run: bool = False) -> bool:
        self._ocr_results = None

        try:
            # Step 1: Identify media
            self._stage('identify')
            logger.info(f'Identifying: {video_path.name}')

            # Step 2: Extract reference track for validation
            self._stage('extract')
            reference_path = self._extract_reference(video_path, work_dir)

            # Step 3: Search, download, and validate subtitle candidates
            self._stage('fetch')
            fetched = self._search_and_validate(video_path, work_dir, reference_path)
            fetched_eng = fetched.get('eng')
            fetched_pol = fetched.get('pol')

            if fetched_pol:
                self._stage('fetch', 'Polish validated')
                logger.info('Using validated Polish subtitles')
            elif fetched_eng:
                self._stage('fetch', 'English validated')
                logger.info('Using validated English subtitles')
            else:
                self._stage('fetch', 'none found')

            # Step 4: Determine English subtitle source
            if fetched_eng:
                extracted_ass = fetched_eng
            elif reference_path:
                extracted_ass = reference_path
            else:
                # No fetched English and no reference — try extraction from scratch
                extracted_ass = self._extract_subtitles(video_path, work_dir)
                if not extracted_ass:
                    return False

            # Step 5: Translate (if needed)
            if fetched_pol:
                logger.info('Using fetched Polish — skipping translation')
                polish_ass = fetched_pol
                dialogue_lines = SubtitleProcessor.extract_dialogue_lines(extracted_ass)
                if not dialogue_lines:
                    logger.error('No dialogue lines found')
                    return False
                translated_dialogue = None
                self._stage('translate', 'skipped')
            else:
                self._stage('translate')
                dialogue_lines = SubtitleProcessor.extract_dialogue_lines(extracted_ass)
                if not dialogue_lines:
                    logger.error('No dialogue lines found')
                    return False

                logger.info(f'Translating {len(dialogue_lines)} lines...')
                try:
                    translated_dialogue = translate_dialogue_lines(
                        dialogue_lines, self.device, self.batch_size, self.model
                    )
                    if not translated_dialogue:
                        logger.error('Translation failed')
                        return False
                except Exception as e:
                    logger.error(f'Translation failed: {e}')
                    return False
                polish_ass = None

            # Step 6: Create subtitle files
            self._stage('create')
            fonts_support_polish = check_embedded_fonts_support_polish(video_path, extracted_ass)

            clean_english_ass = work_dir / f'{video_path.stem}_english_clean.ass'
            SubtitleProcessor.create_english_subtitles(
                extracted_ass, dialogue_lines, clean_english_ass
            )
            SubtitleProcessor.validate_cleaned_subtitles(extracted_ass, clean_english_ass)

            if polish_ass is None:
                polish_ass = work_dir / f'{video_path.stem}_polish.ass'
                replace_chars = not fonts_support_polish
                SubtitleProcessor.create_polish_subtitles(
                    extracted_ass, translated_dialogue, polish_ass, replace_chars
                )

            # Step 6.5: Inpaint burned-in subtitles if detected
            source_video = video_path
            if self._ocr_results:
                logger.info('Removing burned-in subtitles...')
                inpainted_video = work_dir / f'{video_path.stem}_inpainted{video_path.suffix}'
                remove_burned_in_subtitles(
                    video_path, inpainted_video, self._ocr_results, self.device
                )
                source_video = inpainted_video

            # Step 7: Mux final video
            self._stage('mux')
            temp_video = work_dir / f'{video_path.stem}_temp{video_path.suffix}'
            video_ops = self._get_video_ops()
            video_ops.create_clean_video(source_video, clean_english_ass, polish_ass, temp_video)
            video_ops.verify_result(temp_video)

            if not dry_run:
                self._replace_original(video_path, temp_video)

            return True

        except Exception as e:
            logger.error(f'Failed: {video_path.name} - {e}')
            return False

    def _get_extractor(self) -> SubtitleExtractor:
        if self._extractor is None:
            self._extractor = SubtitleExtractor()
        return self._extractor

    def _get_video_ops(self) -> VideoOperations:
        if self._video_ops is None:
            self._video_ops = VideoOperations()
        return self._video_ops

    def _extract_subtitles(self, video_path: Path, output_dir: Path) -> Path | None:
        extractor = self._get_extractor()

        track_info = extractor.get_track_info(video_path)
        if not track_info:
            logger.error('Could not read track information')
            return None

        eng_track = extractor.find_english_track(track_info)
        if not eng_track:
            if not is_vision_ocr_available():
                logger.error('No English subtitle track found (OCR not available)')
                return None
            if not probe_for_burned_in_subtitles(video_path):
                logger.info('No burned-in subtitles detected — skipping OCR')
                return None
            return self._extract_burned_in_subtitles(video_path, output_dir)

        track_id = eng_track['id']
        codec = eng_track.get('codec_name', '?')
        logger.info(f'English track: {codec} (ID {track_id})')

        subtitle_ext = extractor.get_subtitle_extension(eng_track)
        extracted_sub = output_dir / f'{video_path.stem}_extracted{subtitle_ext}'
        subtitle_index = eng_track.get('subtitle_index', 0)
        extractor.extract_subtitle(video_path, track_id, extracted_sub, subtitle_index)

        return extracted_sub

    def _extract_burned_in_subtitles(self, video_path: Path, output_dir: Path) -> Path | None:
        logger.info('Attempting burned-in subtitle OCR...')
        result = extract_burned_in_subtitles(video_path, output_dir)
        if result is None:
            return None
        self._ocr_results = result.ocr_results
        return result.srt_path

    def _replace_original(self, video_path: Path, temp_video: Path) -> None:
        backup_path = video_path.with_suffix(video_path.suffix + '.backup')
        shutil.copy2(video_path, backup_path)

        try:
            shutil.move(str(temp_video), str(video_path))
            video_ops = self._get_video_ops()
            video_ops.verify_result(video_path)
            backup_path.unlink()
        except Exception:
            if backup_path.exists() and not video_path.exists():
                shutil.move(str(backup_path), str(video_path))
            raise
```

- [ ] **Step 2: Update main.py to use new working directory flow**

Edit `movie_translator/main.py`. Replace `find_video_files_with_temp_dirs` and update the main loop:

```python
def find_video_files(input_dir: Path) -> list[Path]:
    """Find all video files in input directory (flat or one level deep)."""
    video_files: list[Path] = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(input_dir.glob(ext))

    if not video_files:
        for subdir in sorted(input_dir.iterdir()):
            if subdir.is_dir() and not subdir.name.startswith('.'):
                for ext in VIDEO_EXTENSIONS:
                    video_files.extend(subdir.glob(ext))

    video_files.sort()
    return video_files
```

Update the `main()` function to use `create_working_dirs`, pass `work_dir` to `process_video_file`, and handle cleanup:

```python
def main():
    args = parse_args()
    set_verbose(args.verbose)

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        console.print(f'[red]❌ Directory not found: {input_dir}[/red]')
        sys.exit(1)

    if not check_dependencies():
        sys.exit(1)

    video_files = find_video_files(input_dir)

    if not video_files:
        console.print(f'[red]❌ No video files found in {input_dir}[/red]')
        sys.exit(1)

    total_files = len(video_files)
    keep_artifacts = args.keep_artifacts

    if args.dry_run:
        console.print('[yellow]Dry run mode - originals will not be modified[/yellow]')

    logging.getLogger('transformers').setLevel(logging.ERROR)

    extractor = SubtitleExtractor()

    with ProgressTracker(total_files, console=console) as tracker:
        pipeline = TranslationPipeline(
            device=args.device,
            batch_size=args.batch_size,
            model=args.model,
            enable_fetch=not args.no_fetch,
            tracker=tracker,
        )

        for video_path in video_files:
            relative_name = (
                f'{video_path.parent.name}/{video_path.name}'
                if video_path.parent != input_dir
                else video_path.name
            )

            tracker.start_file(relative_name)
            work_dir = create_working_dirs(video_path, input_dir)

            try:
                if extractor.has_polish_subtitles(video_path):
                    tracker.complete_file('skipped')
                    success = True
                elif pipeline.process_video_file(video_path, work_dir, dry_run=args.dry_run):
                    tracker.complete_file('success')
                    success = True
                else:
                    tracker.complete_file('failed')
                    success = False
            except Exception as e:
                logger.error(f'Unexpected error: {e}')
                tracker.complete_file('failed')
                success = False

            # Clean up working directory (keep on failure or if flag set)
            if success and not keep_artifacts:
                import shutil
                shutil.rmtree(work_dir, ignore_errors=True)
                # Clean up empty parent dirs
                anime_dir = work_dir.parent
                if anime_dir.exists() and not any(anime_dir.iterdir()):
                    anime_dir.rmdir()
                temp_root = anime_dir.parent
                if temp_root.exists() and temp_root.name == '.translate_temp' and not any(temp_root.iterdir()):
                    temp_root.rmdir()
```

- [ ] **Step 3: Run all tests**

Run: `cd /Users/w/h_dev/movie_translator && python -m pytest movie_translator/ -v --ignore=.venv`
Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add movie_translator/pipeline.py movie_translator/main.py
git commit -m "feat: integrate subtitle validation into pipeline with working directories"
```

---

### Task 8: Empirical Calibration with Test Data

**Files:**
- Create: `scripts/calibrate_validator.py`

This task creates a standalone script that runs the fingerprinting algorithm against the real anime test data to determine optimal bin size and threshold values.

- [ ] **Step 1: Create calibration script**

```python
#!/usr/bin/env python3
"""Calibrate subtitle validator against real anime test data.

For each video file with an embedded English subtitle track:
1. Extract the embedded subtitle as reference
2. Search providers for all candidates
3. Score each candidate against the reference
4. Also cross-test: score each episode's candidates against OTHER episodes' references

This produces a score matrix to determine optimal threshold and bin size.

Usage:
    python scripts/calibrate_validator.py ~/Downloads/translated

Requires OPENSUBTITLES_API_KEY environment variable.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from movie_translator.identifier import identify_media
from movie_translator.logging import logger, set_verbose
from movie_translator.subtitle_fetch import SubtitleFetcher
from movie_translator.subtitle_fetch.providers.animesub import AnimeSubProvider
from movie_translator.subtitle_fetch.providers.opensubtitles import OpenSubtitlesProvider
from movie_translator.subtitle_fetch.validator import (
    SubtitleValidator,
    build_activity_vector,
    compute_similarity,
    extract_timestamps,
)
from movie_translator.subtitles import SubtitleExtractor


def find_videos(input_dir: Path) -> list[Path]:
    videos = []
    for ext in ('*.mkv', '*.mp4'):
        videos.extend(input_dir.rglob(ext))
    return sorted(videos)


def main():
    parser = argparse.ArgumentParser(description='Calibrate subtitle validator')
    parser.add_argument('input_dir', help='Directory with anime video files')
    parser.add_argument('--bin-sizes', default='1000,2000,5000,10000', help='Bin sizes to test (ms)')
    parser.add_argument('--max-videos', type=int, default=10, help='Max videos to process')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    set_verbose(args.verbose)
    input_dir = Path(args.input_dir)
    bin_sizes = [int(x) for x in args.bin_sizes.split(',')]

    videos = find_videos(input_dir)[:args.max_videos]
    print(f'Found {len(videos)} videos (processing up to {args.max_videos})')

    extractor = SubtitleExtractor()
    providers = [AnimeSubProvider()]
    api_key = os.environ.get('OPENSUBTITLES_API_KEY', '')
    if api_key:
        providers.append(OpenSubtitlesProvider(api_key=api_key))
    fetcher = SubtitleFetcher(providers)

    # Phase 1: Extract references and collect candidates
    references: dict[str, Path] = {}  # video_name -> ref_path
    ref_timestamps: dict[str, list[tuple[int, int]]] = {}
    candidates: dict[str, list[tuple[str, Path]]] = {}  # video_name -> [(name, path)]

    with tempfile.TemporaryDirectory(prefix='calibrate_') as tmpdir:
        tmp = Path(tmpdir)

        for video in videos:
            name = video.stem
            print(f'\n--- {name} ---')

            # Extract reference
            track_info = extractor.get_track_info(video)
            if not track_info:
                print(f'  SKIP: no track info')
                continue

            eng_track = extractor.find_english_track(track_info)
            if not eng_track:
                print(f'  SKIP: no English track')
                continue

            ref_dir = tmp / name
            ref_dir.mkdir(parents=True, exist_ok=True)

            ref_path = ref_dir / f'ref{extractor.get_subtitle_extension(eng_track)}'
            subtitle_index = eng_track.get('subtitle_index', 0)
            extractor.extract_subtitle(video, eng_track['id'], ref_path, subtitle_index)

            ts, dur = extract_timestamps(ref_path)
            if not ts:
                print(f'  SKIP: no dialogue in reference')
                continue

            references[name] = ref_path
            ref_timestamps[name] = ts
            print(f'  Reference: {len(ts)} dialogue events, {dur}ms')

            # Search for candidates
            try:
                identity = identify_media(video)
                matches = fetcher.search_all(identity, ['eng', 'pol'])
                print(f'  Candidates: {len(matches)} found')

                video_candidates = []
                for i, match in enumerate(matches[:10]):  # limit to 10 per video
                    cand_path = ref_dir / f'cand_{i}_{match.source}_{match.language}.{match.format}'
                    try:
                        fetcher.download_candidate(match, cand_path)
                        video_candidates.append((f'{match.release_name} ({match.source}/{match.language})', cand_path))
                    except Exception as e:
                        print(f'  Download failed: {match.release_name}: {e}')

                candidates[name] = video_candidates
            except Exception as e:
                print(f'  Search failed: {e}')
                candidates[name] = []

        # Phase 2: Score matrix
        print('\n\n=== SCORE MATRIX ===\n')

        for bin_size in bin_sizes:
            print(f'\n--- Bin size: {bin_size}ms ---')

            for ref_name, ref_path in references.items():
                ref_ts, ref_dur = ref_timestamps[ref_name]

                # Score this episode's own candidates
                print(f'\n  {ref_name}:')
                for cand_name, cand_path in candidates.get(ref_name, []):
                    cand_ts, cand_dur = extract_timestamps(cand_path)
                    if not cand_ts:
                        print(f'    {cand_name}: NO DIALOGUE')
                        continue

                    duration = max(ref_dur, cand_dur)
                    ref_vec = build_activity_vector(ref_ts, duration, bin_size)
                    cand_vec = build_activity_vector(cand_ts, duration, bin_size)
                    score = compute_similarity(ref_vec, cand_vec)
                    print(f'    OWN  | {score:.3f} | {cand_name}')

                # Cross-test: score against other episodes' candidates
                for other_name in list(references.keys())[:5]:
                    if other_name == ref_name:
                        continue
                    for cand_name, cand_path in candidates.get(other_name, [])[:3]:
                        cand_ts, cand_dur = extract_timestamps(cand_path)
                        if not cand_ts:
                            continue
                        duration = max(ref_dur, cand_dur)
                        ref_vec = build_activity_vector(ref_ts, duration, bin_size)
                        cand_vec = build_activity_vector(cand_ts, duration, bin_size)
                        score = compute_similarity(ref_vec, cand_vec)
                        print(f'    CROSS| {score:.3f} | {other_name} candidate: {cand_name}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run calibration against test data**

Run: `cd /Users/w/h_dev/movie_translator && python scripts/calibrate_validator.py ~/Downloads/translated --max-videos 6 -v`

Observe the score matrix. Expected pattern:
- OWN candidates: high scores (0.6-0.95)
- CROSS candidates (wrong episode): low scores (0.05-0.30)

- [ ] **Step 3: Update threshold based on empirical results**

Based on the calibration results, update the default `min_threshold` value in `SubtitleValidator.validate_candidates()` in `movie_translator/subtitle_fetch/validator.py`. If the gap between correct and incorrect scores is clear, pick a threshold in the middle of that gap.

- [ ] **Step 4: Update bin_size default if needed**

If a bin size other than 2000ms produces better separation, update the default `bin_size_ms` parameter in `SubtitleValidator.__init__()`.

- [ ] **Step 5: Commit**

```bash
git add scripts/calibrate_validator.py movie_translator/subtitle_fetch/validator.py
git commit -m "feat: add calibration script and tune validator thresholds from empirical data"
```

---

### Task 9: End-to-End Smoke Test

**Files:** None created — this is a manual verification step.

- [ ] **Step 1: Run full pipeline on one anime episode**

Pick an episode from the test data that previously got wrong subtitles (KonoSuba):

Run: `cd /Users/w/h_dev/movie_translator && python -m movie_translator "$HOME/Downloads/translated/[Cerberus] KonoSuba S1 + S2 + OVA + Movie [BD 1080p HEVC 10-bit OPUS] [Dual-Audio]/Kono Subarashii Sekai ni Shukufuku wo!" --dry-run --keep-artifacts -v`

Verify:
- Working directories created under `.translate_temp/<anime>/<episode>/`
- Reference subtitle extracted
- Multiple candidates downloaded
- Validation scores logged
- Best candidate selected (or none if all fail validation)
- Artifacts preserved (due to `--keep-artifacts`)

- [ ] **Step 2: Inspect artifacts**

Check the working directory contents:
Run: `ls -R "$HOME/Downloads/translated/[Cerberus] KonoSuba S1 + S2 + OVA + Movie [BD 1080p HEVC 10-bit OPUS] [Dual-Audio]/Kono Subarashii Sekai ni Shukufuku wo!/.translate_temp/"`

Verify candidates/ and reference/ subdirectories exist with expected files.

- [ ] **Step 3: Run on a clean anime (Aho-Girl) to verify normal flow**

Run: `cd /Users/w/h_dev/movie_translator && python -m movie_translator "$HOME/Downloads/translated/[smplstc] Aho-Girl (BD 1080p x265 10-Bit Opus)" --dry-run --keep-artifacts -v`

Verify the pipeline completes successfully with validation.

- [ ] **Step 4: Commit any final fixes**

If any issues were discovered during smoke testing, fix and commit them.

```bash
git add -A
git commit -m "fix: address issues discovered during end-to-end smoke testing"
```

---

### Summary of Task Dependencies

```
Task 1 (activity vector) → Task 2 (cross-correlation) → Task 3 (timestamp parsing) → Task 4 (SubtitleValidator class)
Task 5 (fetcher refactor) — independent of Tasks 1-4 until Task 7
Task 6 (working dirs) — independent of Tasks 1-5
Task 7 (pipeline integration) — depends on Tasks 4, 5, 6
Task 8 (calibration) — depends on Task 4
Task 9 (smoke test) — depends on Task 7
```

Tasks 1-4 are sequential (each builds on the previous).
Tasks 5 and 6 can run in parallel with each other and with Tasks 1-4.
Task 7 requires all of Tasks 4, 5, 6 to be complete.
Task 8 can start after Task 4 (doesn't need pipeline integration).
