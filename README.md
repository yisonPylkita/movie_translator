# Movie Translator 🎬

**AI subtitle translator** for English→Polish translation. Works on macOS and Linux.

## Features

- **💻 Cross-Platform** - Works on macOS (Intel/Apple Silicon) and Linux
- **🍎 MPS Acceleration** - Optimized for Apple Silicon (M1/M2/M3)
- **🤖 AI Translation** - High-quality models (Allegro BiDi, mBART)
- **🎥 Multi-Format** - Supports MKV, MP4, AVI, WebM, MOV
- ** Smart Filtering** - Extracts dialogue only (skips signs/songs)
- **📦 Zero System Dependencies** - FFmpeg bundled via Python, no Homebrew needed

## Requirements

- **macOS** or **Linux**
- **[uv](https://docs.astral.sh/uv/)** - Python package manager
- That's it! FFmpeg is bundled automatically.

## Quick Start

### Setup (One-time)

```bash
# Option 1: Use the setup script
./setup.sh

# Option 2: Manual setup with uv
uv sync
```

### Usage

```bash
# Translate video files in a directory
./run.sh ~/Downloads/movies

# Use mBART model instead of default Allegro
./run.sh ~/Downloads/movies --model mbart

# Adjust batch size for memory/speed tradeoff
./run.sh ~/Downloads/movies --batch-size 8

# Use CPU instead of MPS
./run.sh ~/Downloads/movies --device cpu

# Show all options
./run.sh --help
```

### With OCR Support (Optional)

```bash
# Install OCR dependencies
uv sync --extra ocr

# Process image-based subtitles
./run.sh ~/Downloads/movies --enable-ocr
```

## UV Commands Reference

```bash
# Install/sync dependencies
uv sync

# Run the translator
uv run movie-translator <input_dir> [options]

# Run with dev dependencies
uv sync --group dev

# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Run tests
uv run pytest
```

## Supported Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| Matroska | `.mkv` | Full support |
| MP4 | `.mp4` | Full support |
| AVI | `.avi` | Full support |
| WebM | `.webm` | Full support |
| QuickTime | `.mov` | Full support |

## Project Structure

```
movie_translator/
├── pyproject.toml           # UV/Python project configuration
├── uv.lock                  # Locked dependencies
├── setup.sh                 # Setup convenience script
├── run.sh                   # Run convenience script
└── movie_translator/        # Main package
    ├── main.py              # CLI entry point
    ├── pipeline.py          # Translation pipeline
    ├── ffmpeg.py            # FFmpeg utilities (bundled)
    ├── utils.py             # Logging utilities
    ├── translation/         # AI translation module
    ├── subtitles/           # Subtitle processing
    ├── video/               # Video operations
    └── ocr/                 # OCR support (optional)
```

## Translation Models

| Model | Key | Description |
|-------|-----|-------------|
| Allegro BiDi | `allegro` | English-Polish (default, recommended) |
| mBART | `mbart` | Multilingual (50 languages) |

## How It Works

1. **Extract** - English subtitles from video file
2. **Filter** - Dialogue only (no signs/songs)
3. **Translate** - AI translation with progress bar
4. **Create** - Clean video with Polish + English tracks

## License

MIT License - see LICENSE file for details.
