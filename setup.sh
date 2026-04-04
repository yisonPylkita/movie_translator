#!/bin/bash
set -e

# One-time setup for new contributors
echo "🎬 Setting up Movie Translator..."

# Install system dependencies
brew bundle

# Enable direnv for this directory
direnv allow

# Initialize submodules (ilass alignment engine)
git submodule update --init --recursive

# Fetch AI translation model
git lfs install
git lfs pull

MODEL_FILE="models/allegro/model.safetensors"
MODEL_SIZE=$(stat -f%z "$MODEL_FILE" 2>/dev/null || stat -c%s "$MODEL_FILE" 2>/dev/null || echo "0")
if [[ "$MODEL_SIZE" -gt 1000000 ]]; then
    echo "✅ Model ready ($((MODEL_SIZE / 1024 / 1024))MB)"
else
    echo "❌ Failed to download model via Git LFS"
    exit 1
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Usage:"
echo "  just run ~/Downloads/movies               # translate MKV/MP4 files"
echo "  just run ~/Downloads/movies --dry-run      # preview without modifying"
echo "  just run ~/Downloads/movies --enable-ocr   # OCR burned-in subtitles"
echo "  just --list                                # see all dev commands"
