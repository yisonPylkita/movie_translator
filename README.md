# Movie Translator

[![Tests](https://github.com/yisonPylkita/movie_translator/actions/workflows/tests.yml/badge.svg)](https://github.com/yisonPylkita/movie_translator/actions/workflows/tests.yml)

**AI subtitle translator** for English to Polish translation. Runs entirely locally on your machine.

## What It Does

Takes video files (MKV or MP4) with English subtitles and produces new video files with both Polish (AI-translated) and English subtitle tracks. Polish is set as the default track.

Supports two subtitle sources:
- **Subtitle tracks** (ASS, SRT) embedded in MKV/MP4 files
- **Burned-in subtitles** (hardcoded into video frames) via Apple Vision OCR (macOS only)

## Requirements

- **macOS** (recommended) or **Linux**
- **[uv](https://docs.astral.sh/uv/)** - Python package manager
- That's it! FFmpeg is bundled automatically.

For burned-in subtitle OCR, macOS with Apple Silicon is required (uses the Neural Engine).

## Quick Start

### 1. Setup

```bash
# Clone and enter the repo
git clone https://github.com/yisonPylkita/movie_translator.git
cd movie_translator

# Option A: Automated setup (macOS only - installs direnv, just, git-lfs via Homebrew)
./setup.sh

# Option B: Manual setup
uv sync              # Install Python dependencies
git lfs install      # Enable Git LFS
git lfs pull         # Download the translation model (~446MB)
```

### 2. Translate Videos

```bash
# Translate all MKV/MP4 files in a directory
just run ~/Downloads/movies

# Or without just:
uv run movie-translator ~/Downloads/movies
```

The tool will:
1. Find all `.mkv` and `.mp4` files in the directory (and one level of subdirectories)
2. Skip files that already have Polish subtitles
3. Extract English subtitles, translate to Polish, and mux both tracks back into the video

### 3. Common Options

```bash
# Preview without modifying originals (output goes to .translate_temp/)
just run ~/Downloads/movies --dry-run

# Adjust batch size for memory/speed tradeoff
just run ~/Downloads/movies --batch-size 8

# Use CPU instead of Apple Silicon GPU
just run ~/Downloads/movies --device cpu

# Verbose logging
just run ~/Downloads/movies --verbose

# Show all options
just run -- --help
```

### 4. Burned-In Subtitle OCR (macOS only)

For videos with subtitles baked into the video frames (no subtitle tracks), use OCR extraction:

```bash
# Install OCR dependencies
uv sync --extra vision-ocr

# Process videos with burned-in subtitles
just run ~/Downloads/movies --enable-ocr
```

This uses Apple's Vision framework on the Neural Engine. It extracts frames at 1fps, OCRs the subtitle region (bottom 25% of the frame), and produces an SRT file that feeds into the translation pipeline. Typical speed: ~80 seconds for a 30-minute video.

## How It Works

```
Video file (MKV/MP4)
  -> Extract English subtitles (from tracks or via OCR)
  -> Filter dialogue (skip signs/songs/effects)
  -> Translate to Polish (Allegro BiDi model, runs on MPS/CPU)
  -> Check if embedded fonts support Polish characters
  -> Create Polish + clean English subtitle files
  -> Mux both tracks into the video (Polish as default)
```

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run all checks and tests (CI equivalent)
just ci

# Individual commands
just lint       # Auto-fix linting and formatting
just check      # Check without modifying files (lint + format + type check)
just test       # Run tests
just run        # Run the CLI
```

### Project Structure

```
movie_translator/
  main.py              # CLI entry point, file discovery
  pipeline.py          # Translation orchestration
  ffmpeg.py            # FFmpeg/FFprobe wrapper
  types.py             # Core data types (DialogueLine, SubtitleFile)
  fonts.py             # Font analysis for Polish character support
  subtitles/           # Subtitle extraction and processing
  translation/         # AI translation (Allegro BiDi model)
  video/               # Video muxing and verification
  ocr/                 # Burned-in subtitle OCR (Apple Vision)
```

## License

MIT License - see LICENSE file for details.
