# Movie Translator 🎬

[![Tests](https://github.com/yisonPylkita/movie_translator/actions/workflows/tests.yml/badge.svg)](https://github.com/yisonPylkita/movie_translator/actions/workflows/tests.yml)

**AI subtitle translator** for English→Polish translation. Works on macOS and Linux.

## Features

- **💻 Cross-Platform** - Works on macOS and Linux
- **🍎 MPS Acceleration** - Optimized for Apple Silicon
- **🤖 AI Translation** - Runs locally using pre-verified translation models
- **🎯 Smart Filtering** - Extracts dialogue only (skips signs/songs)
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

## License

MIT License - see LICENSE file for details.
