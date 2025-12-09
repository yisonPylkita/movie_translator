#!/bin/bash
# Setup script for Movie Translator
# This is a convenience wrapper around uv commands

set -e

echo "🍎 Setting up Movie Translator for MacBook..."
echo ""

# Check macOS
if [[ "$(uname)" != "Darwin" ]]; then
	echo "❌ This setup is designed for MacBook only"
	exit 1
fi

if [[ "$(uname -m)" == "arm64" ]]; then
	echo "✅ Apple Silicon MacBook detected"
else
	echo "⚠️  Intel Mac detected - MPS acceleration not available"
fi

# Install uv if needed
if ! command -v uv &>/dev/null; then
	echo "📦 Installing uv..."
	curl -LsSf https://astral.sh/uv/install.sh | sh
	export PATH="$HOME/.cargo/bin:$PATH"
fi
echo "✅ uv $(uv --version)"

# Install mkvtoolnix if needed
if ! command -v mkvmerge &>/dev/null; then
	echo "📦 Installing mkvtoolnix..."
	if ! command -v brew &>/dev/null; then
		/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
	fi
	brew install mkvtoolnix
fi
echo "✅ mkvtoolnix installed"

# Sync dependencies with uv
echo "📦 Syncing Python dependencies..."
uv sync

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Usage:"
echo "  uv run movie-translator ~/Downloads/movies"
echo "  uv run movie-translator --help"
echo ""
echo "With OCR support:"
echo "  uv sync --extra ocr"
echo "  uv run movie-translator --enable-ocr ~/Downloads/movies"
