# Subtitle Validation via Timing Pattern Fingerprinting

## Problem

Downloaded subtitles from external databases (OpenSubtitles, AnimeSub) may be from the wrong episode despite matching the correct show. The current pipeline blindly trusts the best-scored result. This was discovered with KonoSuba where the downloaded subtitle was from the right anime but wrong episode.

## Solution Overview

Validate downloaded subtitles against a reference track (embedded English subtitle or OCR-extracted timing) using **timing pattern fingerprinting**. Instead of matching individual lines, compare the pattern of speech activity over time — treating subtitle timing as a binary signal and using cross-correlation to score similarity.

Additionally, instead of downloading only the single best match, **download all plausible candidates** and score them locally, selecting the best validated match.

Introduce **per-video working directories** for all intermediate artifacts, with cleanup by default and a `--keep-artifacts` flag.

## Architecture

### 1. Working Directory Structure

```
input_dir/
  .translate_temp/
    Aho-Girl/                          # per-anime parent
      Aho-Girl - 01/                   # per-episode working dir
        candidates/                    # all downloaded subtitle candidates
          opensubtitles_eng_12345.srt
          opensubtitles_eng_67890.srt
          animesub_pol_abc.ass
        reference/                     # extracted embedded/OCR subs
          embedded_eng.ass
        selected/                      # winning subtitle after validation
          best_eng.srt
          best_pol.ass
      Aho-Girl - 02/
        ...
    Undead Unluck/
      [EMBER] Undead Unluck - 01/
        ...
```

- The per-anime parent directory is derived from the video's parent folder name
- The per-episode directory uses the video file stem
- Default behavior: cleaned up after each video completes successfully
- `--keep-artifacts` CLI flag: preserves working directories for debugging
- Failed videos always preserve their working directory regardless of flag

### 2. Timing Pattern Fingerprinting Algorithm

The core insight: the temporal pattern of "when people talk" is a fingerprint unique to each episode. Even with timing offsets, extra lines, or format differences, the overall pattern is recognizable.

#### Step-by-step:

1. **Parse subtitles** — extract start/end timestamps from both reference and candidate (using pysubs2, format-agnostic)
2. **Filter to dialogue** — remove non-dialogue events (signs, songs, effects) using existing `NON_DIALOGUE_STYLES` filtering
3. **Build activity vectors** — divide the timeline into fixed-width bins (e.g., 2-second windows). For each bin, set 1 if any dialogue overlaps it, 0 otherwise. This produces a binary vector per subtitle file.
4. **Cross-correlate** — compute normalized cross-correlation between the two vectors. This finds the best alignment even if one subtitle is consistently shifted. The correlation at the best offset is the similarity score.
5. **Score** — the peak correlation value (0.0 to 1.0) represents how well the timing patterns match.

#### Why this works:

- **Slight timing offsets**: cross-correlation finds the best alignment automatically
- **Extra lines (OP/ED, signs)**: filtered out in step 2; remaining extras add minor noise but don't destroy the pattern
- **Format differences (SRT vs ASS)**: only timing is used, not formatting
- **Cross-language**: only measures *when* dialogue happens, not *what* is said
- **Wrong episode**: completely different dialogue distribution = low correlation

#### Bin size selection:

The bin size controls granularity vs noise tolerance. Too small (100ms) and minor timing differences cause mismatches. Too large (30s) and different episodes may look similar.

Starting point: **2-second bins**. This will be validated empirically against the test data (5 anime series, ~70 episodes). The implementation should make bin size configurable for experimentation.

#### Threshold:

The threshold for "plausibly correct episode" will be determined empirically. Expected ranges:
- Correct episode: 0.7-0.95 correlation
- Wrong episode: 0.05-0.30 correlation

The implementation will include a research/calibration mode that outputs scores for all candidates against all references to help determine the optimal threshold. The initial threshold will be set conservatively (e.g., 0.5) and tuned based on real data.

### 3. Reference Track Hierarchy

The ground truth for validation, in priority order:

1. **Embedded text subtitle track** — extracted via FFmpeg, best quality timing data
2. **OCR-extracted subtitles** — from burned-in subtitle detection, usable as timing cue
3. **No reference available** — log a warning, skip local validation, rely on provider scores only

### 4. Fetcher Changes

Current behavior: `SubtitleFetcher.fetch_subtitles()` picks the single best match per language and downloads it.

New behavior:

#### SubtitleFetcher API changes:

```python
class SubtitleFetcher:
    def search_all(
        self,
        identity: MediaIdentity,
        languages: list[str],
    ) -> list[SubtitleMatch]:
        """Search all providers, return all plausible candidates."""
        ...

    def download_candidate(
        self,
        match: SubtitleMatch,
        output_path: Path,
    ) -> Path:
        """Download a single candidate subtitle."""
        ...
```

- `search_all()` replaces the search-and-download-best logic
- Returns ALL matches (not just best per language)
- Caller handles download decisions based on validation results
- Provider-level scoring still happens (hash_match, score field) but is no longer the sole selection criterion

#### SubtitleMatch changes:

Add a `download_url` or equivalent field so downloads can happen separately from search. (Check if providers already expose this — OpenSubtitles has a download endpoint by ID, AnimeSub has a URL.)

### 5. Validation Module

New module: `movie_translator/subtitle_fetch/validator.py`

```python
class SubtitleValidator:
    """Validates downloaded subtitles against a reference track."""

    def __init__(self, reference_path: Path, bin_size_ms: int = 2000):
        """Load and fingerprint the reference subtitle."""
        ...

    def score_candidate(self, candidate_path: Path) -> float:
        """Score a candidate subtitle against the reference. Returns 0.0-1.0."""
        ...

    def validate_candidates(
        self,
        candidates: list[tuple[SubtitleMatch, Path]],
        min_threshold: float = 0.5,
    ) -> list[tuple[SubtitleMatch, Path, float]]:
        """Score and rank all candidates. Returns sorted by validation score, filtered by threshold."""
        ...
```

Key implementation details:
- Uses pysubs2 for parsing (already a dependency)
- Filters non-dialogue events before fingerprinting
- Uses numpy for cross-correlation (already available via torch dependency)
- Returns scores alongside matches for logging/debugging

### 6. Pipeline Integration

The pipeline flow changes from:

```
identify → fetch best → extract/use → translate → mux
```

To:

```
identify → extract reference → search all candidates → download all
→ validate each against reference → select best per language
→ translate (if needed) → mux
```

Detailed flow in `pipeline.py`:

1. **Extract reference track** (moved earlier in pipeline)
   - Try embedded English subtitle extraction
   - Fall back to OCR if no embedded track
   - If neither available, set `reference = None`

2. **Search all providers** for candidates
   - `fetcher.search_all(identity, ['eng', 'pol'])`

3. **Download all candidates** into `working_dir/candidates/`

4. **Validate if reference exists**
   - Create `SubtitleValidator(reference_path)`
   - Score each candidate
   - Rank by validation score
   - Filter by threshold

5. **Select best per language**
   - If validation was possible: use validation scores
   - If no reference: fall back to provider scores (current behavior + warning log)
   - If no candidate passes threshold: skip this video (log error)

6. **Continue with selected subtitles** (translate, mux, etc.)

### 7. CLI Changes

New flags in `main.py`:

- `--keep-artifacts`: Preserve working directories after processing (default: clean up on success)

### 8. Logging

- Log all candidate scores during validation (at INFO level for the selected one, DEBUG for others)
- Log which reference source was used (embedded/OCR/none)
- Warn when no reference is available for validation
- Log when a candidate is rejected due to low validation score

## Test Plan

### Empirical Calibration (using ~/Downloads/translated)

1. For each anime in the test set:
   - Extract the embedded English subtitle from each episode
   - Download all available candidates from providers
   - Score each candidate against each episode's reference (including deliberate cross-episode comparisons)
   - Record the score matrix: correct episode should have high scores, wrong episodes should have low scores
2. Use the score matrix to determine:
   - Optimal bin size (try 1s, 2s, 5s, 10s)
   - Optimal threshold (find the value that best separates correct from incorrect)

### Unit Tests

- Fingerprinting: test with known subtitle files (identical, shifted, different content)
- Cross-correlation: test with synthetic binary vectors
- Candidate selection: test ranking and threshold filtering
- Working directory: test creation, cleanup, and --keep-artifacts behavior

### Integration Tests

- Full pipeline run with the test anime data
- Verify correct episode subtitles are selected
- Verify wrong episode subtitles are rejected

## Files to Create/Modify

### New Files
- `movie_translator/subtitle_fetch/validator.py` — fingerprinting and validation logic

### Modified Files
- `movie_translator/subtitle_fetch/fetcher.py` — split into search_all + download_candidate
- `movie_translator/subtitle_fetch/types.py` — add download info to SubtitleMatch if needed
- `movie_translator/pipeline.py` — new validation flow, working directory management, reference extraction moved earlier
- `movie_translator/main.py` — `--keep-artifacts` flag, updated working directory logic

## Out of Scope

- Audio-based validation (matching subtitle timing to audio activity)
- Content-based matching (comparing actual subtitle text across languages)
- Automatic subtitle re-timing/shifting (if close but offset)
- Caching downloaded candidates across runs
