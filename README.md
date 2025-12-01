# Movie Translator 🎬

**MacBook-only AI subtitle translator** for English→Polish translation using Apple Silicon acceleration.

> ⚠️ **Important**: This project is designed and optimized for MacBook only. For other systems, please create a separate setup.

## Features 🚀

- **🍎 MacBook Optimized**: MPS acceleration for Apple Silicon (M1/M2/M3)
- **🤖 AI Translation**: High-quality `allegro/BiDi-eng-pol` model
- **🎯 Smart Filtering**: Extracts dialogue only (skips signs/songs)
- **🎨 Rich Progress**: Beautiful terminal output with live speed metrics
- **🧠 Memory Efficient**: Proven leak-free implementation

## System Requirements 🍎

- **macOS** with Apple Silicon (M1/M2/M3) recommended
- **Python 3.10+** (managed by uv)
- **Homebrew** (for mkvtoolnix)
- **uv** (Python package manager)

> **Intel Macs**: May work but without MPS acceleration (slower performance)
> **Non-Mac systems**: Not supported - please create separate setup

## Quick Start 🎯

### Setup (One-time)
```bash
cd /Users/arlen/h_dev/movie_translator
./setup.sh  # Install dependencies and check requirements
```

### Usage
```bash
# Process directory (uses MPS by default)
uv run python translate.py ~/Downloads/test_movies

# Process single file
uv run python translate.py ~/Downloads/test_movies/SPY\ x\ FAMILY\ -\ S01E01.mkv

# Custom output
uv run python translate.py ~/Downloads/test_movies --output ~/Downloads/translated
```

### Advanced Options
```bash
# Different batch sizes
uv run python translate.py ~/Downloads/test_movies --batch-size 32  # Faster
uv run python translate.py ~/Downloads/test_movies --batch-size 8   # Less memory

# CPU fallback (if MPS issues)
uv run python translate.py ~/Downloads/test_movies --device cpu
```

## Project Structure 📁

```
movie_translator/
├── translate.py          # Main pipeline (MacBook optimized)
├── ai_translator.py      # AI engine (MPS acceleration)
├── run.sh               # Quick test script
├── setup.sh             # Setup script (dependencies + requirements)
├── pyproject.toml       # uv dependencies
└── README.md            # This file
```

## MacBook Optimization 🍎

This project is **exclusively optimized for MacBook**:

- **Default Device**: MPS (Apple Silicon GPU) - fastest performance
- **Batch Size**: 16 (optimized for MacBook memory)
- **Memory Management**: Leak-free with periodic cleanup
- **Dependencies**: uv-managed for consistency
- **Setup Detection**: Automatically verifies MacBook compatibility

### Platform Support
- ✅ **Apple Silicon Macs** (M1/M2/M3) - Full support
- ⚠️ **Intel Macs** - Works but slower (no MPS)
- ❌ **Windows/Linux** - Not supported (create separate setup)

## Dependencies 📦

MacBook-optimized dependencies managed by `uv`:
- `torch` (MPS acceleration for Apple Silicon)
- `transformers` (AI models)
- `rich` (terminal UI)
- `pysubs2` (subtitle handling)

**System Requirements:**
- macOS with Apple Silicon (M1/M2/M3)
- `mkvtoolnix` (`brew install mkvtoolnix`)

## How It Works 🔄

1. **Extract** English subtitles from MKV
2. **Filter** dialogue only (no signs/songs)
3. **Translate** with AI + Rich progress bar
4. **Create** clean MKV with English + Polish tracks

## License 📄

MIT License - see LICENSE file for details.
