# Movie Translator 🎬

**MacBook-only AI subtitle translator** for English→Polish translation using Apple Silicon acceleration.

> ⚠️ **Important**: This project is designed and optimized for MacBook only.

## Features

- **🍎 MacBook Optimized** - MPS acceleration for Apple Silicon (M1/M2/M3)
- **🤖 AI Translation** - High-quality models (Allegro BiDi, mBART)
- **🎯 Smart Filtering** - Extracts dialogue only (skips signs/songs)
- **🎨 Rich Progress** - Beautiful terminal output with live speed metrics
- **📦 UV-First** - Modern Python tooling with `uv`

## Requirements

- **macOS** with Apple Silicon (M1/M2/M3) recommended
- **[uv](https://docs.astral.sh/uv/)** - Python package manager
- **mkvtoolnix** - `brew install mkvtoolnix`

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
# Translate MKV files in a directory
uv run movie-translator ~/Downloads/movies

# Use mBART model instead of default Allegro
uv run movie-translator ~/Downloads/movies --model mbart

# Adjust batch size for memory/speed tradeoff
uv run movie-translator ~/Downloads/movies --batch-size 8

# Use CPU instead of MPS
uv run movie-translator ~/Downloads/movies --device cpu

# Show all options
uv run movie-translator --help
```

### With OCR Support (Optional)

```bash
# Install OCR dependencies
uv sync --extra ocr

# Process image-based subtitles
uv run movie-translator ~/Downloads/movies --enable-ocr
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
    ├── translation/         # AI translation module
    ├── subtitles/           # Subtitle processing
    ├── mkv/                 # MKV operations
    └── ocr/                 # OCR support (optional)
```

## Translation Models

| Model | Key | Description |
|-------|-----|-------------|
| Allegro BiDi | `allegro` | English-Polish (default, recommended) |
| mBART | `mbart` | Multilingual (50 languages) |

## How It Works

1. **Extract** - English subtitles from MKV
2. **Filter** - Dialogue only (no signs/songs)
3. **Translate** - AI translation with progress bar
4. **Create** - Clean MKV with Polish + English tracks

## License

MIT License - see LICENSE file for details.
