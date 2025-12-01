#!/bin/bash

set -e # Exit on any error

echo "🎬 Running Movie Translator Test..."
echo ""

cleanup_test_directory() {
	echo "🧹 Cleaning test directory..."
	rm -rf ~/Downloads/test_movies/*
	echo "   ✅ Test directory cleaned"
}

copy_test_files() {
	echo "📁 Copying test files..."
	local source_dir="$HOME/Downloads/Torrents/completed/[neoDESU] SPY x FAMILY [Season 1+2] [BD 1080p x265 HEVC OPUS AAC] [Dual Audio]/Season 1"
	
	cp "$source_dir/SPY x FAMILY - S01E01.mkv" ~/Downloads/test_movies/
	cp "$source_dir/SPY x FAMILY - S01E02.mkv" ~/Downloads/test_movies/
	cp "$source_dir/SPY x FAMILY - S01E03.mkv" ~/Downloads/test_movies/
	echo "   ✅ 3 test files copied"
}

run_translation() {
	echo "🚀 Running translation..."
	uv run python translate.py ~/Downloads/test_movies
}

show_completion() {
	echo ""
	echo "🎉 Test complete!"
	echo "Check ~/Downloads/test_movies/translated/ for results."
}

# Main test flow
main() {
	cleanup_test_directory
	copy_test_files
	run_translation
	show_completion
}

main "$@"
