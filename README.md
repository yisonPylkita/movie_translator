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

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run linter
uv run ruff check .

# Run type checker
uv run ty check

# Run formatter
uv run ruff format .

# Run tests
uv run pytest -v
```

## License

MIT License - see LICENSE file for details.


❯ ./run.sh ~/Downloads/test_movies 
🎬 Movie Translator

DEBUG    Attempting to acquire lock 4401263488 on /Users/arlen/h_dev/movie_translator/.venv/lib/python3.14/site-packages/static_ffmpeg/lock.file                                                          
DEBUG    Lock 4401263488 acquired on /Users/arlen/h_dev/movie_translator/.venv/lib/python3.14/site-packages/static_ffmpeg/lock.file                                                                       
DEBUG    Attempting to release lock 4401263488 on /Users/arlen/h_dev/movie_translator/.venv/lib/python3.14/site-packages/static_ffmpeg/lock.file                                                          
DEBUG    Lock 4401263488 released on /Users/arlen/h_dev/movie_translator/.venv/lib/python3.14/site-packages/static_ffmpeg/lock.file                                                                       
🎬 Movie Translator - 1 file(s)
INFO     Processing: Jujutsu Kaisen 0.mkv                                                                                                                                                                 
INFO     Extracting subtitles...                                                                                                                                                                          
INFO     Found 2 English subtitle track(s):                                                                                                                                                               
INFO       Track 1: ID=3, Name="Signs", Codec=ass                                                                                                                                                         
INFO       Track 2: ID=4, Name="Full Subtitles", Codec=ass                                                                                                                                                
INFO     Categorized: 0 dialogue track(s), 2 signs/songs track(s)                                                                                                                                         
ERROR    Only signs/songs tracks available (2 track(s)). Full dialogue track required. Skipping this video.                                                                                               
ERROR    No English subtitle track found   