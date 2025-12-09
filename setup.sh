#!/bin/bash
set -e

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

echo "📦 Syncing Python dependencies..."
uv sync

echo ""
echo "🎉 Setup complete!"
echo ""
echo "All dependencies installed via Python - no system packages required!"
echo "FFmpeg is bundled via static-ffmpeg package."
echo ""
echo "Usage:"
echo "  ./run.sh ~/Downloads/movies"
echo ""
echo "Supported format: MKV"
echo ""
echo "Options:"
echo "  ./run.sh ~/Downloads/movies --model facebook"
echo "  ./run.sh ~/Downloads/movies --batch-size 8"
echo "  ./run.sh --help"
echo ""
echo "With OCR support:"
echo "  uv sync --extra ocr"
echo "  ./run.sh ~/Downloads/movies --enable-ocr"
