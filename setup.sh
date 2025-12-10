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

if ! command -v git-lfs &>/dev/null; then
	if [[ "$OS" == "Darwin" ]] && command -v brew &>/dev/null; then
		echo "📦 Installing Git LFS..."
		brew install git-lfs
	else
		echo "❌ Git LFS is required but not installed"
		echo ""
		echo "   Install it with:"
		echo "   - macOS: brew install git-lfs"
		echo "   - Ubuntu/Debian: sudo apt install git-lfs"
		echo "   - Fedora: sudo dnf install git-lfs"
		exit 1
	fi
fi
echo "✅ Git LFS found"
git lfs install >/dev/null 2>&1

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
echo "🤖 Fetching AI translation model..."
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
