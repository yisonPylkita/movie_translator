#!/bin/bash

set -e # Exit on any error

cleanup_previous_run() {
	echo "🧹 Cleaning previous run results..."
	rm -rf ~/Downloads/test_movies/*
	echo "   ✅ Previous run results cleaned"
}

copy_test_files() {
	echo "📁 Copying test files..."
	local source_dir="$HOME/Downloads/Torrents/completed/[neoDESU] SPY x FAMILY [Season 1+2] [BD 1080p x265 HEVC OPUS AAC] [Dual Audio]/Season 1"

	# cp "$source_dir/SPY x FAMILY - S01E01.mkv" ~/Downloads/test_movies/
	cp "$source_dir/SPY x FAMILY - S01E02.mkv" ~/Downloads/test_movies/
	# cp "$source_dir/SPY x FAMILY - S01E03.mkv" ~/Downloads/test_movies/
	echo "   ✅ Test files copied"
}

run_translation() {
	echo "🚀 Running translation..."
	uv run python translate.py ~/Downloads/test_movies
}

show_completion() {
	echo ""
	echo "🎉 Translation complete!"
	echo "Check ~/Downloads/test_movies/translated/ for results."
}

echo "🎬 Running Movie Translator"
echo ""

main() {
	cleanup_previous_run
	copy_test_files
	run_translation
	show_completion
}

main "$@"
