#!/bin/bash
# Run script for Movie Translator
# This is a convenience wrapper around: uv run movie-translator

set -e

echo "🎬 Movie Translator"
echo ""

# Pass all arguments to uv run movie-translator
if [[ $# -eq 0 ]]; then
	echo "Usage: ./run.sh <input_directory> [options]"
	echo ""
	echo "Examples:"
	echo "  ./run.sh ~/Downloads/movies"
	echo "  ./run.sh ~/Downloads/movies --model mbart"
	echo "  ./run.sh ~/Downloads/movies --batch-size 8"
	echo ""
	echo "For all options:"
	echo "  uv run movie-translator --help"
	exit 0
fi

uv run movie-translator "$@"
