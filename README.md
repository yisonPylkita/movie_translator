# Movie Translator

[![Tests](https://github.com/yisonPylkita/movie_translator/actions/workflows/tests.yml/badge.svg)](https://github.com/yisonPylkita/movie_translator/actions/workflows/tests.yml)

**AI subtitle translator** for English to Polish translation. Fetches existing Polish subtitles from the internet when available, falls back to AI translation, and handles timing alignment automatically. Runs entirely locally on your machine.

## What It Does

Takes video files (MKV or MP4) and produces new video files with as many Polish subtitle tracks as possible. The pipeline is fully automated:

1. **Identifies** the media (title, season, episode) from the filename
2. **Extracts** the English subtitle track from the video (text or OCR for burned-in/PGS)
3. **Searches** multiple subtitle providers for existing Polish translations
4. **Validates** downloaded subtitles against the English reference (timing-based scoring)
5. **Aligns** fetched subtitles to the video's timeline (handles OP removal, different video cuts)
6. **Translates** with AI as a fallback when no Polish subtitles are found online
7. **Muxes** everything back into the video — all Polish tracks plus the original English

There is also a standalone **extract** command for pulling subtitles out of videos (text tracks and burned-in OCR) without translating.

### Subtitle Sources

All available sources are included as separate tracks in the output:

- **Internet downloads** from AnimeSub, Podnapisi, NapiProjekt, and OpenSubtitles
- **AI translation** using the Allegro BiDi model or Apple Translation (macOS 26+)
- **External pre-extracted subtitles** via `--external-subs` (from a prior `extract` run)

### Subtitle Alignment

Fetched subtitles are often timed to different video releases. The alignment system handles:

- **Small offsets** (1-3s) from different video encode start points
- **Large offsets** (60-90s+) from OP/ED removal in the subtitle source video
- **Piecewise alignment** with different offsets for pre-OP and post-OP segments

Primary alignment uses [ilass](https://github.com/SandroHc/ilass) (DP algorithm with split penalties), with a built-in cross-correlation fallback.

## Requirements

- **macOS** (recommended) or **Linux**
- **[uv](https://docs.astral.sh/uv/)** — Python package manager
- **Rust toolchain** (for building ilass alignment engine)
- **FFmpeg** development libraries (for ilass audio-based alignment)

For burned-in subtitle OCR, macOS with Apple Silicon is required (uses Apple Vision framework).

## Quick Start

### 1. Setup

```bash
git clone --recurse-submodules https://github.com/yisonPylkita/movie_translator.git
cd movie_translator

# Install Python dependencies
uv sync

# Build ilass alignment engine (requires Rust + FFmpeg dev libs)
# macOS: brew install pkg-config ffmpeg
cd vendor/ilass && cargo build --release && cd ../..
```

> **Already cloned without `--recurse-submodules`?** Run `git submodule update --init --recursive` to fetch the ilass submodule.

### 2. Translate Videos

```bash
# Translate all MKV/MP4 files in a directory
uv run movie-translator ~/Downloads/anime

# Or with just:
just run ~/Downloads/anime
```

The tool will:
1. Find all `.mkv` and `.mp4` files recursively
2. Skip files that already have Polish subtitles
3. Search for Polish subtitles online, validate and align them
4. Fall back to AI translation if no suitable match is found
5. Mux all Polish + English tracks into the video

### 3. Extract Subtitles

Extract subtitles from videos without translating — useful for pulling burned-in Polish subtitles from a low-quality source to apply to a better version:

```bash
# Extract burned-in Polish subtitles via OCR
uv run movie-translator extract ~/Downloads/polish_version --ocr-language pl

# Then use them when translating the high-quality version
uv run movie-translator ~/Downloads/english_version --external-subs ~/Downloads/polish_version/extracted_subs
```

The extract command outputs SRT files and a `manifest.json` that the translate command uses for matching by media identity (title + season + episode).

### 4. Common Options

```bash
# Preview without modifying originals
uv run movie-translator ~/Downloads/anime --dry-run

# Disable online subtitle fetching (AI-only)
uv run movie-translator ~/Downloads/anime --no-fetch

# Use Apple Translation backend (macOS 26+)
uv run movie-translator ~/Downloads/anime --model apple

# Process multiple files concurrently
uv run movie-translator ~/Downloads/anime --workers 4

# Adjust batch size for memory/speed tradeoff
uv run movie-translator ~/Downloads/anime --batch-size 8

# Use CPU instead of Apple Silicon GPU
uv run movie-translator ~/Downloads/anime --device cpu

# Show all options
uv run movie-translator --help
uv run movie-translator extract --help
```

## How It Works

```
Video file (MKV/MP4)
  |
  +-> Identify media (title, season, episode, file hashes)
  +-> Extract English subtitle track (ASS/SRT/PGS via FFmpeg, or OCR for burned-in)
  |
  +-> Search subtitle providers (AnimeSub, Podnapisi, NapiProjekt, OpenSubtitles)
  +-> Download all Polish candidates
  +-> Validate candidates against English reference (line-level timing match)
  +-> Select best candidates (keep multiple if score >= 0.8)
  +-> Align to video timeline (ilass DP alignment / cross-correlation fallback)
  |
  +-> AI translation fallback (Allegro BiDi or Apple Translation)
  +-> Check if embedded fonts support Polish characters
  +-> Add external pre-extracted subtitles (if --external-subs provided)
  +-> Create subtitle tracks (fetched Polish + AI Polish + external + English)
  +-> Mux into video (Polish as default track)
```

### Burned-In Subtitle OCR

For videos with hardcoded subtitles (no subtitle streams), the OCR pipeline:

1. Extracts the bottom 25% of video frames at 3 FPS, scaled to 1280px width
2. Runs pixel-diff change detection to identify frames where subtitles changed
3. OCRs only the transition frames using Apple Vision (typically ~28% of total frames)
4. Deduplicates consecutive identical text and builds timed SRT output

### Dialogue Detection

ASS subtitle files often contain thousands of non-dialogue events (karaoke, signs, typesetting). The structural classifier identifies dialogue styles based on event properties rather than style name keywords:

- **Positioning ratio** — signs and karaoke use explicit `\pos()`/`\move()`
- **Text length** — karaoke syllables are 1-3 characters
- **Event density** — karaoke has many rapid-fire short events

This approach works across arbitrary fansub naming conventions without maintaining a keyword list.

## Development

```bash
uv sync --group dev

just ci          # Run all checks and tests (CI equivalent)
just lint        # Auto-fix linting and formatting
just check       # Check without modifying (lint + format + type check)
just test        # Run tests
just run <dir>   # Run the translate CLI
```

## License

MIT License - see LICENSE file for details.
