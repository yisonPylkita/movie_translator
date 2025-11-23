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

### Option 1: Automatic (Recommended - macOS)

The `translate.sh` script automatically installs everything you need!

```bash
# 1. Clone the repository
git clone git@github.com:yisonPylkita/movie_translator.git
cd movie_translator

# 2. Run the script - it handles all dependencies automatically
./translate.sh /path/to/your/movies
```

**What it installs automatically:**
- ✅ Homebrew (if not present)
- ✅ uv (Python package manager)
- ✅ mkvtoolnix (for MKV file manipulation)

**Requirements:**
- Python 3 (pre-installed on macOS)
- Terminal

That's it! No manual setup required.

### Option 2: Manual Installation

If you prefer to install dependencies manually:

```bash
# 1. Install mkvtoolnix
brew install mkvtoolnix

# 2. Install uv (Python package manager)
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
