# Movie Translator 🎬

**Translate movie/anime subtitles from English to Polish** using local ML models.

Optimized for **M1/M2 MacBook Air** - runs completely offline, no API keys needed.

## Three Tools, One Purpose

### 1. `srt-translate` - Pure SRT Translation
Translate any SRT file (input → output):
```bash
srt-translate input.srt output.srt
```

### 2. `movie-translate` - Full MKV Workflow
Extract → Translate → Merge subtitles in MKV files:
```bash
movie-translate /path/to/movies
```

### 3. `srt-validate` - Subtitle Validation
Validate that translated subtitles match the source in structure and timing:
```bash
srt-validate source.srt translated.srt
```

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

### Tool 1: SRT Translation Only

Translate standalone SRT files:

```bash
# Basic translation
uv run srt-translate input.srt output.srt

# With options
uv run srt-translate input.srt output.srt --device cpu --batch-size 32
```

**Use when:** You already have extracted SRT files, or want to translate subtitles from any source.

### Tool 2: Full MKV Processing

Automatic extraction, translation, and merging:

```bash
# Process all MKV files in a directory
uv run movie-translate /path/to/movies

# Process single MKV file
uv run movie-translate movie.mkv

# With options
uv run movie-translate /path/to/movies --device auto --batch-size 16
```

**Use when:** You have MKV files with embedded English subtitles and want automated processing.

### Tool 3: Subtitle Validation

Verify that translated subtitles match the source structure:

```bash
# Validate English vs Polish subtitles
uv run srt-validate subtitles/en_full.srt subtitles/pl_full.srt
```

**What it checks:**
- ✅ Number of subtitle entries matches
- ✅ Start/end timestamps match (within 50ms tolerance)
- ✅ Duration consistency
- ⚠️ HTML formatting tags preserved (`<i>`, `<b>`, `<u>`)
- ⚠️ Line breaks maintained

**Use when:** You want to ensure translation didn't break subtitle timing or structure, especially useful for QA or debugging translation issues.

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
