#!/bin/bash

set -e # Exit on any error

ensure_system_macos() {
	echo "🍎 Checking system compatibility..."
	if [[ "$(uname)" != "Darwin" ]]; then
		echo "   ❌ This setup is designed for MacBook only"
		echo "   ❌ Current system: $(uname)"
		echo "   💡 For non-MacBook systems, please create a separate setup script"
		exit 1
	fi

	if [[ "$(uname -m)" != "arm64" ]]; then
		echo "   ⚠️  Warning: This is optimized for Apple Silicon (arm64)"
		echo "   ⚠️  Current architecture: $(uname -m)"
		echo "   💡 Intel Macs may work but won't have MPS acceleration"
	else
		echo "   ✅ Apple Silicon MacBook detected"
	fi
}

install_uv() {
	echo "📦 Checking uv..."
	if ! command -v uv &>/dev/null; then
		echo "   Installing uv..."
		curl -LsSf https://astral.sh/uv/install.sh | sh
		export PATH="$HOME/.cargo/bin:$PATH"
		echo "   ✅ uv installed"
	else
		echo "   ✅ uv already installed"
	fi
}

install_homebrew_if_needed() {
	if ! command -v brew &>/dev/null; then
		echo "📦 Installing Homebrew..."
		/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
		echo "   ✅ Homebrew installed"
	else
		echo "   ✅ Homebrew already installed"
	fi
}

install_mkvtoolnix() {
	echo "📦 Checking mkvtoolnix..."
	if ! command -v mkvmerge &>/dev/null; then
		install_homebrew_if_needed
		echo "   Installing mkvtoolnix..."
		brew install mkvtoolnix
		echo "   ✅ mkvtoolnix installed"
	else
		echo "   ✅ mkvtoolnix already installed"
	fi
}

install_python_dependencies() {
	echo "📦 Installing Python dependencies..."
	uv sync
	echo "   ✅ Dependencies installed"
}

show_usage() {
	echo ""
	echo "🎉 Setup complete!"
	echo ""
	echo "Quick usage:"
	echo "  uv run python translate.py ~/Downloads/test_movies"
	echo ""
	echo "For more options:"
	echo "  uv run python translate.py --help"
}

# Main setup flow
main() {
	echo "🍎 Setting up Movie Translator for MacBook..."
	echo ""

	ensure_system_macos
	install_uv
	install_mkvtoolnix
	install_python_dependencies
	show_usage
}

main "$@"
