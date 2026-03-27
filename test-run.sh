#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="/Users/w/Downloads/[101-105] Reverse Mountain [En Sub][1080p]"
WORK_DIR="$(cd "$(dirname "$0")" && pwd)/test_workdir"
MAX_DURATION=300  # 5 minutes

echo "=== Movie Translator Test Run ==="
echo "Input:  $INPUT_DIR"
echo "Work:   $WORK_DIR"
echo "Limit:  ${MAX_DURATION}s (first video only)"
echo

# Clean previous run
if [ -d "$WORK_DIR" ]; then
    echo "Cleaning previous test run..."
    rm -rf "$WORK_DIR"
fi
mkdir -p "$WORK_DIR"

# Pick first video and trim to MAX_DURATION
FIRST_VIDEO="$(ls "$INPUT_DIR"/*.mp4 | head -1)"
BASENAME="$(basename "$FIRST_VIDEO")"
echo "Trimming '$BASENAME' to ${MAX_DURATION}s..."

FFMPEG="$(command -v ffmpeg || echo /opt/homebrew/bin/ffmpeg)"
"$FFMPEG" -y -i "$FIRST_VIDEO" -t "$MAX_DURATION" -c copy "$WORK_DIR/$BASENAME" 2>/dev/null
echo "Created $(du -h "$WORK_DIR/$BASENAME" | cut -f1) test clip"
echo

# Run translator
echo "Running translator with OCR..."
uv run movie-translator "$WORK_DIR" --enable-ocr --verbose

echo
echo "=== Done ==="
echo "Results in: $WORK_DIR"
echo "Open with:  open $WORK_DIR"
