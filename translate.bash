#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./translate.bash extract /path/to/mkv/folder
#   ./translate.bash apply /path/to/mkv/folder

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 [extract|apply] /path/to/mkv/folder"
  exit 1
fi

MODE="$1"
WORKDIR="$2"

if [ ! -d "$WORKDIR" ]; then
  echo "Error: Directory does not exist: $WORKDIR"
  exit 1
fi

cd "$WORKDIR"

# Helper directories (auto-created inside WORKDIR)
ENG_DIR="$WORKDIR/english_subtitles"
POL_DIR="$WORKDIR/polish_subtitles"
OUT_DIR="$WORKDIR/output_with_polish"

mkdir -p "$ENG_DIR" "$POL_DIR" "$OUT_DIR"

if [ "$MODE" = "extract" ]; then
  echo "=== Extracting English full subtitles from MKVs in: $WORKDIR ==="

  shopt -s nullglob
  for f in *.mkv; do
    echo "Processing: $f"
    json=$(mkvmerge -J "$f")

    track_id=$(echo "$json" | jq -r '
      .tracks[]
      | select(.type=="subtitles")
      | select(.properties.language=="eng")
      | select((.properties.track_name | test("(?i)sign|song") | not))
      | .id
    ' | head -n 1)

    if [ -n "$track_id" ]; then
      base="${f%.*}"
      out="$ENG_DIR/${base}_eng_full.srt"
      echo "  Found English subtitle track ID: $track_id"
      mkvextract tracks "$f" "$track_id:$out"
      echo "  Saved to: $out"
    else
      echo "  ⚠️  No full English subtitles found for $f"
    fi
  done

elif [ "$MODE" = "apply" ]; then
  echo "=== Applying Polish subtitles to MKVs in: $WORKDIR ==="

  shopt -s nullglob
  for f in *.mkv; do
    base="${f%.*}"
    polish="$POL_DIR/${base}_eng_full_Polish.srt"

    if [ -f "$polish" ]; then
      out="$OUT_DIR/${base}_pl.mkv"
      echo "Applying: $polish → $out"
      mkvmerge -o "$out" "$f" --language 0:pol "$polish"
    else
      echo "⚠️  No matching Polish subtitles for $f"
    fi
  done

else
  echo "Invalid mode: $MODE"
  echo "Usage: $0 [extract|apply] /path/to/mkv/folder"
  exit 1
fi
