#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🎬 Setting up Movie Translator..."
echo ""

OS="$(uname)"
if [[ "$OS" == "Darwin" ]]; then
	if [[ "$(uname -m)" == "arm64" ]]; then
		echo "✅ Apple Silicon Mac detected (MPS acceleration available)"
	else
		echo "✅ Intel Mac detected"
	fi
elif [[ "$OS" == "Linux" ]]; then
	echo "✅ Linux detected"
else
	echo "⚠️  Unknown OS: $OS - may not be fully supported"
fi

if ! command -v uv &>/dev/null; then
	echo "📦 Installing uv..."
	curl -LsSf https://astral.sh/uv/install.sh | sh
	export PATH="$HOME/.cargo/bin:$PATH"
fi
echo "✅ uv $(uv --version)"

echo ""
echo "📦 Syncing Python dependencies..."
uv sync

echo ""
echo "📦 Downloading FFmpeg (static binary)..."
uv run python -c "from static_ffmpeg import run; run.get_or_fetch_platform_executables_else_raise()" 2>/dev/null
echo "✅ FFmpeg ready"

echo ""
echo "🤖 Checking AI translation model..."
if [[ -f "models/allegro/model.safetensors" ]]; then
	echo "✅ Model already exists"
else
	echo "📥 Downloading model from HuggingFace..."
	uv run python -c "
from huggingface_hub import snapshot_download
from pathlib import Path
import shutil

target = Path('models/allegro')
target.mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id='allegro/BiDi-eng-pol', local_dir=str(target))

# Clean up unnecessary files
for f in target.glob('*.svg'):
    f.unlink()
for f in target.glob('*.md'):
    f.unlink()
cache_dir = target / '.cache'
if cache_dir.exists():
    shutil.rmtree(cache_dir)
gitattr = target / '.gitattributes'
if gitattr.exists():
    gitattr.unlink()

print('✅ Model downloaded successfully')
"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "FFmpeg is bundled via static-ffmpeg package."
echo "AI model is stored in models/allegro directory."
echo ""
echo "Usage:"
echo "  ./run.sh ~/Downloads/movies"
echo ""
echo "Supported format: MKV"
echo ""
echo "Options:"
echo "  ./run.sh ~/Downloads/movies --batch-size 8"
echo "  ./run.sh ~/Downloads/movies --dry-run"
echo "  ./run.sh --help"
echo ""
echo "With OCR support:"
echo "  uv sync --extra ocr"
echo "  ./run.sh ~/Downloads/movies --enable-ocr"
