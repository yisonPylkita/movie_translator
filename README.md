# SRT Translator 🚀

**Simple, fast, local SRT subtitle translator** from English to Polish.

Optimized for **M1/M2 MacBook Air** - runs completely offline using local ML models.

## Features

- ✅ **100% Local** - No API keys, no internet, no cloud
- ⚡ **M1 Optimized** - Uses MPS acceleration + float16 for 3-5x speed
- 🎯 **Single Purpose** - Input SRT → Output SRT, nothing else
- 🪶 **Lightweight** - ~222MB model, 2-4GB RAM
- 🔋 **Fast** - ~50-100 lines/second on M1 with MPS

## Requirements

- **Python 3.13** (3.10-3.13 supported, NOT 3.14)
- `uv` package manager
- M1/M2 Mac (or any system with CPU/CUDA)

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

### Basic Translation

```bash
# Simple: input.srt → output.srt
uv run srt-translate input.srt output.srt
```

### Advanced Options

```bash
# Force CPU (slower but more compatible)
uv run srt-translate input.srt output.srt --device cpu

# Increase batch size for speed (if you have RAM)
uv run srt-translate input.srt output.srt --batch-size 32

# Use different model
uv run srt-translate input.srt output.srt --model allegro/p5-eng2many
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Translation model | `gsarti/opus-mt-tc-en-pl` |
| `--device` | `auto`, `cpu`, `cuda`, or `mps` | `auto` (detects M1) |
| `--batch-size` | Lines per batch (higher = faster + more RAM) | `16` |

## How It Works

1. **Loads** SRT file and extracts subtitle lines
2. **Batches** lines for efficient processing
3. **Translates** using local Marian MT model:
   - Auto-detects M1/MPS acceleration
   - Uses float16 precision for 2x speed
   - Greedy decoding for fastest results
4. **Saves** translated SRT with original timing

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

**One tool, one job:** This tool only does SRT translation.

For a complete subtitle workflow:
1. **Extract** subtitles from MKV: Use `mkvextract`
2. **Translate** SRT file: Use this tool
3. **Merge** back into MKV: Use `mkvmerge`

Decoupling these steps gives you more control and flexibility.

## License

MIT
