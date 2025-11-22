# Setup Guide for M1 MacBook Air

## Critical: Python Version Requirement

**You must use Python 3.10-3.13**. Python 3.14 has a known bug with the `sentencepiece` library that prevents T5 models from loading.

### Check Your Python Version

```bash
python3 --version
```

If you're on Python 3.14, install Python 3.13:

```bash
# Using Homebrew
brew install python@3.13

# Set as default for this project
cd /path/to/movie_translator
echo "3.13" > .python-version
```

Then recreate the virtual environment:

```bash
# Remove old venv
rm -rf .venv

# Reinstall with correct Python version
uv sync
```

## Installation Steps

1. **Install uv** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Install MKVToolNix** (required for processing MKV files):
```bash
brew install mkvtoolnix
```

3. **Sync dependencies**:
```bash
uv sync
```

## M1 Optimization Features

This tool is optimized for maximum speed on M1/M2 MacBooks:

- **MPS Acceleration**: Automatically uses Metal Performance Shaders (Apple Silicon GPU)
- **Float16 Precision**: 2x faster inference on M1 with minimal quality loss
- **Batch Processing**: Processes 16 subtitle lines at once for better throughput
- **Memory Management**: Aggressive garbage collection to prevent memory issues
- **Greedy Decoding**: Fastest generation strategy (num_beams=1)

## Usage

```bash
# Basic usage (auto-detects MPS on M1)
uv run movie-translator /path/to/movies

# Force CPU mode (slower but more compatible)
uv run movie-translator /path/to/movies --device cpu

# Increase batch size for faster processing (if you have enough RAM)
uv run movie-translator /path/to/movies --batch-size 32

# Use different model
uv run movie-translator /path/to/movies --model sdadas/mt5-base-translator-en-pl
```

## Performance Tips

### For Maximum Speed:
1. Use default batch size (16) or higher if you have 16GB+ RAM
2. Let it auto-detect MPS (don't force CPU)
3. Close other applications to free up memory
4. Use the default model (`sdadas/flan-t5-base-translator-en-pl`)

### If You Run Out of Memory:
1. Reduce batch size: `--batch-size 8` or `--batch-size 4`
2. Use CPU mode: `--device cpu`
3. Close other applications

## Expected Performance on M1 Air

- **Model Download**: ~220MB, one-time (cached locally)
- **First Run**: 10-20 seconds to load model into memory
- **Translation Speed**:
  - With MPS + Float16: ~50-100 lines/second
  - CPU only: ~10-20 lines/second
- **Memory Usage**: 2-4GB RAM

## Troubleshooting

### "Failed to load model: not a string"
- **Cause**: Python 3.14 incompatibility with sentencepiece
- **Solution**: Use Python 3.13 or lower (see Python Version Requirement above)

### "mkvmerge: command not found"
- **Solution**: Install MKVToolNix: `brew install mkvtoolnix`

### Model loads but translation is slow
- **Check**: Make sure MPS is being used (you should see "Detected M1/M2 Mac - using MPS acceleration" in logs)
- **Try**: Close other applications to free up GPU resources
- **Alternative**: The tool will still work on CPU, just slower

### Out of memory errors
- **Reduce batch size**: `--batch-size 8` or `--batch-size 4`
- **Use CPU**: `--device cpu` (uses less memory but slower)
