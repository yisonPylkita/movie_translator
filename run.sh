#!/bin/bash
set -e

echo "🎬 Movie Translator"
echo ""

if [[ $# -eq 0 ]]; then
	echo "Usage: ./run.sh <input_directory> [options]"
	echo ""
	echo "Examples:"
	echo "  ./run.sh ~/Downloads/movies"
	echo "  ./run.sh ~/Downloads/movies --model facebook"
	echo "  ./run.sh ~/Downloads/movies --batch-size 8"
	echo ""
	echo "For all options:"
	echo "  uv run movie-translator --help"
	exit 0
fi

uv run movie-translator "$@"
