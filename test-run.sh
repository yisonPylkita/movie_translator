#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="/Users/w/Downloads/[101-105] Reverse Mountain [En Sub][1080p]"
WORK_DIR="$(cd "$(dirname "$0")" && pwd)/test_workdir"

echo "=== Movie Translator Test Run ==="
echo "Input:  $INPUT_DIR"
echo "Work:   $WORK_DIR"
echo

# Clean previous run
if [ -d "$WORK_DIR" ]; then
    echo "Cleaning previous test run..."
    rm -rf "$WORK_DIR"
fi

# Copy test videos
echo "Copying test videos..."
mkdir -p "$WORK_DIR"
cp "$INPUT_DIR"/*.mp4 "$WORK_DIR/"
echo "Copied $(ls "$WORK_DIR"/*.mp4 2>/dev/null | wc -l | tr -d ' ') video(s)"
echo

# Run translator
echo "Running translator with OCR..."
uv run movie-translator "$WORK_DIR" --enable-ocr --verbose

echo
echo "=== Done ==="
echo "Results in: $WORK_DIR"
echo "Open with:  open $WORK_DIR"
