# Movie Translator 🎬

**Translate movie/anime subtitles from English to Polish** using local ML models.

Optimized for **M1/M2 MacBook Air** - runs completely offline, no API keys needed.

## Tools Overview

### Single File Translation
- **`srt-translate`** - Translate a single SRT file

### Three-Step MKV Workflow (Recommended)
- **`srt-extract`** - Extract English subtitles from MKV files
- **`srt-translate-batch`** - Translate multiple SRT files
- **`srt-apply`** - Merge Polish subtitles back into MKV files

### Utilities
- **`srt-validate`** - Validate subtitle timing and structure

## Features

- ✅ **100% Local** - No API keys, no internet, no cloud
- ⚡ **M1 Optimized** - Uses MPS acceleration + float16 for 3-5x speed
- 🎯 **Modular** - Use separately or together
- 🪶 **Lightweight** - ~222MB model, 2-4GB RAM
- 🔋 **Fast** - ~50-100 lines/second on M1 with MPS

## Requirements

**For both tools:**
- **Python 3.13** (3.10-3.13 supported, NOT 3.14)
- `uv` package manager
- M1/M2 Mac (or any system with CPU/CUDA)

**Additional for `movie-translate`:**
- MKVToolNix (`mkvmerge`, `mkvextract`)
  ```bash
  brew install mkvtoolnix
  ```

## Installation

```bash
# 1. Install Python 3.13 (if needed)
brew install python@3.13

# 2. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Clone and sync
git clone git@github.com:yisonPylkita/movie_translator.git
cd movie_translator
uv sync
```

## Usage

### One-Command Solution (Easiest)

Run the complete workflow with a single script that processes files **one at a time**:

```bash
# Full automatic workflow
./translate.bash /path/to/anime

# With backup and keep SRT files
./translate.bash --backup --keep-srt /path/to/movies

# Custom translation settings
./translate.bash --device mps --batch-size 32 /path/to/anime
```

**Why file-by-file processing?**
- 🎬 **Start watching immediately** - Episode 1 is ready while Episode 2 processes
- 🔄 **Resume-friendly** - Stop and resume anytime
- 💾 **Memory efficient** - Processes one file completely before moving to next

**What it does for each file:**
1. Extracts English subtitle → `movie_en.srt`
2. Translates to Polish → `movie_pl.srt` (strips tags, preserves line breaks)
3. Merges into MKV with **ONLY** English + Polish tracks
4. Cleans up SRT files (unless `--keep-srt`)
5. Moves to next file

**Final result:** Each MKV contains only 2 subtitle tracks:
- English (original, default)
- Polish (AI-generated)

### Manual Control (Individual Files)

All commands now work on **single files** for fine-grained control:

```bash
# Step 1: Extract English subtitles from one MKV
uv run srt-extract movie.mkv
# Creates: movie_en.srt

# Step 2: Translate one SRT file to Polish
uv run srt-translate movie_en.srt movie_pl.srt
# Creates: movie_pl.srt

# Step 3: Apply subtitles to one MKV
uv run srt-apply movie.mkv
# Updates: movie.mkv with English + Polish tracks only
```

**Use this when:** You want to process specific files manually or customize the workflow.

**Benefits:**
- 🎯 **Full control** - Review/edit subtitles between steps
- 🔄 **Resume workflow** - Skip completed steps automatically
- 🛡️ **Safe** - Use `--backup` flag to create .bak files
- ⚡ **Fast** - Translate multiple files in one model loading session

**Advanced options:**
```bash
# Extract with specific pattern
uv run srt-extract /path/to/movies

# Translate with custom settings
uv run srt-translate-batch /path/to/movies --device mps --batch-size 32

# Apply with backup
uv run srt-apply /path/to/movies --backup
```

### Single File Translation

Translate a single SRT file:

```bash
# Basic translation
uv run srt-translate input.srt output.srt

# With options
uv run srt-translate input.srt output.srt --device cpu --batch-size 32
```

### Subtitle Validation

Verify translation quality:

```bash
# Validate timing and structure
uv run srt-validate movie_en.srt movie_pl.srt
```

**Validation checks:**
- ✅ Entry count matches
- ✅ Timestamps match (50ms tolerance)
- ✅ Duration consistency
- ⚠️ Line break preservation (warnings only)

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Translation model | `gsarti/opus-mt-tc-en-pl` |
| `--device` | `auto`, `cpu`, `cuda`, or `mps` | `auto` (detects M1) |
| `--batch-size` | Lines per batch (higher = faster + more RAM) | `16` |

## How It Works

### `srt-translate` (Simple)

1. **Loads** SRT file
2. **Batches** subtitle lines
3. **Translates** using local Marian MT model
4. **Saves** translated SRT with original timing

### `movie-translate` (Full Pipeline)

1. **Scans** for MKV files
2. **Identifies** English subtitle tracks using `mkvmerge`
3. **Extracts** subtitles to SRT using `mkvextract`
4. **Translates** SRT using `srt-translate` logic
5. **Merges** Polish subtitles back into MKV using `mkvmerge`
6. **Replaces** original file with updated version

**Translation engine:**
- Auto-detects M1/MPS acceleration
- Uses float16 precision for 2x speed
- Greedy decoding for fastest results

## Performance

**M1 MacBook Air (MPS + float16):**
- First run: ~10-15s (model loading)
- Translation: ~50-100 lines/second
- Memory: 2-4GB RAM

**CPU mode:**
- Translation: ~10-20 lines/second
- Slower but more compatible

## Troubleshooting

### "Failed to load model: not a string"

**Cause:** Python 3.14 incompatibility with sentencepiece

**Fix:**
```bash
brew install python@3.13
rm -rf .venv
uv sync
```

### Model loads but translation is slow

- Check MPS is detected: Look for "Detected M1/M2 Mac - using MPS acceleration" in logs
- Close other apps to free GPU
- CPU fallback is slower but always works

### Out of memory

- Reduce batch size: `--batch-size 8` or `--batch-size 4`
- Use CPU mode: `--device cpu` (uses less memory)

## Model

Currently using **`gsarti/opus-mt-tc-en-pl`**:
- Marian MT architecture (no sentencepiece issues)
- 222MB model size
- Trained on EN→PL translation
- Good quality for general subtitles

## Design Philosophy

**Separation of concerns:**

- **`srt-translate`**: Core translation logic - reusable, testable, simple
- **`movie-translate`**: MKV orchestration - wraps translation with extraction/merging

**Benefits:**
- Use `srt-translate` for any SRT source (downloads, OCR, etc.)
- Use `movie-translate` for convenience with MKV files
- Both share the same optimized translation engine
- Can be used independently or together

**Example workflow combinations:**
```bash
# Option 1: Full automatic (MKV)
movie-translate movies/

# Option 2: Manual control (extract + translate + merge yourself)
mkvextract tracks movie.mkv 2:eng.srt
srt-translate eng.srt pol.srt
mkvmerge -o output.mkv movie.mkv --language 0:pl pol.srt

# Option 3: Just translate (from any source)
srt-translate downloaded.srt translated.srt
```

## License

MIT




❯ ./translate.sh ~/Downloads/test_movies

==========================================
  Movie Translator - Sequential Workflow
==========================================

[INFO] Directory: /Users/arlen/Downloads/test_movies
[INFO] Device: auto
[INFO] Batch Size: 16
[INFO] Backup: disabled
[INFO] Keep SRT files: false

[INFO] Found 2 MKV file(s)


==========================================
\033[1m[1/2] Processing: SPY x FAMILY - S01E01.mkv\033[0m
==========================================

[INFO] Step 1/3: Extracting English subtitles...
Using CPython 3.13.9 interpreter at: /opt/homebrew/opt/python@3.13/bin/python3.13
Creating virtual environment at: .venv
      Built movie-translator @ file:///Users/arlen/h_dev/movie_translator
Installed 30 packages in 413ms
2025-11-23 21:32:16,796 - INFO - Extracting subtitles from: SPY x FAMILY - S01E01.mkv
2025-11-23 21:32:16,925 - INFO - Extracting subtitle track 3 from SPY x FAMILY - S01E01.mkv...
2025-11-23 21:32:17,094 - INFO -   → Saved to SPY x FAMILY - S01E01_en.srt
2025-11-23 21:32:17,094 - INFO - ✓ Extraction complete: SPY x FAMILY - S01E01_en.srt
[SUCCESS] Extraction complete

[INFO] Step 2/3: Translating to Polish...
2025-11-23 21:32:17,530 - INFO - Translating: SPY x FAMILY - S01E01_en.srt → SPY x FAMILY - S01E01_pl.srt
2025-11-23 21:32:17,530 - INFO - Model: allegro/BiDi-eng-pol
2025-11-23 21:32:17,530 - INFO - Device: auto
2025-11-23 21:32:17,530 - INFO - Batch size: 16
2025-11-23 21:32:17,564 - INFO - Loading subtitles from /Users/arlen/Downloads/test_movies/SPY x FAMILY - S01E01_en.srt
2025-11-23 21:32:24,529 - INFO - Detected M1/M2/M3 Mac - using MPS acceleration
2025-11-23 21:32:31,347 - INFO - Loading model allegro/BiDi-eng-pol on mps...
tokenizer_config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 812/812 [00:00<00:00, 1.25MB/s]
source.spm: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 816k/816k [00:00<00:00, 9.10MB/s]
vocab.json: 804kB [00:00, 39.9MB/s]
special_tokens_map.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 416/416 [00:00<00:00, 1.45MB/s]
/Users/arlen/h_dev/movie_translator/.venv/lib/python3.13/site-packages/transformers/models/marian/tokenization_marian.py:175: UserWarning: Recommended: pip install sacremoses.
  warnings.warn("Recommended: pip install sacremoses.")
config.json: 1.07kB [00:00, 2.27MB/s]
model.safetensors: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 837M/837M [00:55<00:00, 15.0MB/s]
generation_config.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 281/281 [00:00<00:00, 1.50MB/s]
2025-11-23 21:33:33,618 - INFO - M1 optimization: Using float16 precision for 2x speed
2025-11-23 21:33:33,618 - INFO - Model loaded successfully on mps
2025-11-23 21:33:33,618 - INFO - Using precision: float16
2025-11-23 21:33:33,618 - INFO - Translating 6373 subtitle lines in batches of 16...
2025-11-23 21:33:33,618 - INFO - Device: mps, Precision: float16
Translating:   0%|                                                                                                                                                            | 0/6373 [00:00<?, ?lines/s]The following generation flags are not valid and may be ignored: ['early_stopping']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
Translating: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6373/6373 [03:31<00:00, 30.10lines/s]
2025-11-23 21:37:05,339 - INFO - Saving translation to /Users/arlen/Downloads/test_movies/SPY x FAMILY - S01E01_pl.srt
2025-11-23 21:37:05,377 - INFO - Translation complete! Saved to /Users/arlen/Downloads/test_movies/SPY x FAMILY - S01E01_pl.srt
2025-11-23 21:37:05,381 - INFO - ✓ Translation complete: /Users/arlen/Downloads/test_movies/SPY x FAMILY - S01E01_pl.srt
[SUCCESS] Translation complete

[INFO] Step 3/3: Applying subtitles to MKV...
2025-11-23 21:37:06,355 - INFO - Applying subtitles to: SPY x FAMILY - S01E01.mkv
2025-11-23 21:37:06,355 - WARNING - Backup mode: OFF (original file will be overwritten)
2025-11-23 21:37:06,356 - INFO - Merging subtitles into SPY x FAMILY - S01E01.mkv...
2025-11-23 21:37:06,356 - INFO -   → Keeping only English and Polish subtitle tracks
2025-11-23 21:37:07,304 - ERROR - Apply failed: Failed to merge subtitles into SPY x FAMILY - S01E01.mkv. Is mkvmerge installed?
[ERROR] Apply failed for SPY x FAMILY - S01E01.mkv

==========================================
\033[1m[2/2] Processing: SPY x FAMILY - S01E02.mkv\033[0m
==========================================

[INFO] Step 1/3: Extracting English subtitles...
2025-11-23 21:37:07,371 - INFO - Extracting subtitles from: SPY x FAMILY - S01E02.mkv
2025-11-23 21:37:07,435 - INFO - Extracting subtitle track 3 from SPY x FAMILY - S01E02.mkv...
2025-11-23 21:37:07,586 - INFO -   → Saved to SPY x FAMILY - S01E02_en.srt
2025-11-23 21:37:07,586 - INFO - ✓ Extraction complete: SPY x FAMILY - S01E02_en.srt
[SUCCESS] Extraction complete

[INFO] Step 2/3: Translating to Polish...
2025-11-23 21:37:07,649 - INFO - Translating: SPY x FAMILY - S01E02_en.srt → SPY x FAMILY - S01E02_pl.srt
2025-11-23 21:37:07,649 - INFO - Model: allegro/BiDi-eng-pol
2025-11-23 21:37:07,649 - INFO - Device: auto
2025-11-23 21:37:07,649 - INFO - Batch size: 16
2025-11-23 21:37:07,664 - INFO - Loading subtitles from /Users/arlen/Downloads/test_movies/SPY x FAMILY - S01E02_en.srt
2025-11-23 21:37:08,323 - INFO - Detected M1/M2/M3 Mac - using MPS acceleration
2025-11-23 21:37:09,705 - INFO - Loading model allegro/BiDi-eng-pol on mps...
/Users/arlen/h_dev/movie_translator/.venv/lib/python3.13/site-packages/transformers/models/marian/tokenization_marian.py:175: UserWarning: Recommended: pip install sacremoses.
  warnings.warn("Recommended: pip install sacremoses.")
2025-11-23 21:37:13,641 - INFO - M1 optimization: Using float16 precision for 2x speed
2025-11-23 21:37:13,641 - INFO - Model loaded successfully on mps
2025-11-23 21:37:13,642 - INFO - Using precision: float16
2025-11-23 21:37:13,642 - INFO - Translating 2370 subtitle lines in batches of 16...
2025-11-23 21:37:13,642 - INFO - Device: mps, Precision: float16
Translating:   0%|                                                                                                                                                            | 0/2370 [00:00<?, ?lines/s]The following generation flags are not valid and may be ignored: ['early_stopping']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
Translating: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2370/2370 [02:00<00:00, 19.65lines/s]
2025-11-23 21:39:14,255 - INFO - Saving translation to /Users/arlen/Downloads/test_movies/SPY x FAMILY - S01E02_pl.srt
2025-11-23 21:39:14,271 - INFO - Translation complete! Saved to /Users/arlen/Downloads/test_movies/SPY x FAMILY - S01E02_pl.srt
2025-11-23 21:39:14,274 - INFO - ✓ Translation complete: /Users/arlen/Downloads/test_movies/SPY x FAMILY - S01E02_pl.srt
[SUCCESS] Translation complete

[INFO] Step 3/3: Applying subtitles to MKV...
2025-11-23 21:39:14,752 - INFO - Applying subtitles to: SPY x FAMILY - S01E02.mkv
2025-11-23 21:39:14,752 - WARNING - Backup mode: OFF (original file will be overwritten)
2025-11-23 21:39:14,752 - INFO - Merging subtitles into SPY x FAMILY - S01E02.mkv...
2025-11-23 21:39:14,752 - INFO -   → Keeping only English and Polish subtitle tracks
2025-11-23 21:39:15,470 - ERROR - Apply failed: Failed to merge subtitles into SPY x FAMILY - S01E02.mkv. Is mkvmerge installed?
[ERROR] Apply failed for SPY x FAMILY - S01E02.mkv

==========================================
  🎉 Workflow Complete!
==========================================

[SUCCESS] All files processed!

[INFO] All processed MKV files now have:
  ✅ English subtitle (original, default)
  ✅ Polish subtitle (AI-generated)

[SUCCESS] Done! 🎬
