# Movie Translator

[![Tests](https://github.com/yisonPylkita/movie_translator/actions/workflows/tests.yml/badge.svg)](https://github.com/yisonPylkita/movie_translator/actions/workflows/tests.yml)

**AI subtitle translator** for English to Polish translation. Fetches existing Polish subtitles from the internet when available, falls back to AI translation, and handles timing alignment automatically. Runs entirely locally on your machine.

## What It Does

Takes video files (MKV or MP4) with English subtitles and produces new video files with Polish subtitle tracks. The pipeline is fully automated:

1. **Identifies** the media (title, season, episode) from the filename
2. **Extracts** the English subtitle track from the video
3. **Searches** multiple subtitle providers for existing Polish translations
4. **Validates** downloaded subtitles against the English reference (timing-based scoring)
5. **Aligns** fetched subtitles to the video's timeline (handles OP removal, different video cuts)
6. **Translates** with AI as a fallback when no Polish subtitles are found online
7. **Muxes** everything back into the video with Polish as the default track

### Subtitle Sources (in priority order)

- **Internet downloads** from AnimeSub, Podnapisi, NapiProjekt, and OpenSubtitles
- **AI translation** using the Allegro BiDi model (runs on MPS/CPU)
- Both tracks are included when internet subtitles are available

### Subtitle Alignment

Fetched subtitles are often timed to different video releases. The alignment system handles:

- **Small offsets** (1-3s) from different video encode start points
- **Large offsets** (60-90s+) from OP/ED removal in the subtitle source video
- **Piecewise alignment** with different offsets for pre-OP and post-OP segments

Primary alignment uses [ilass](https://github.com/SandroHc/ilass) (DP algorithm with split penalties), with a built-in cross-correlation fallback.

## Requirements

- **macOS** (recommended) or **Linux**
- **[uv](https://docs.astral.sh/uv/)** - Python package manager
- **Rust toolchain** (for building ilass alignment engine)
- **FFmpeg** development libraries (for ilass audio-based alignment)

For burned-in subtitle OCR, macOS with Apple Silicon is required.

## Quick Start

### 1. Setup

```bash
git clone https://github.com/yisonPylkita/movie_translator.git
cd movie_translator

# Install Python dependencies
uv sync

# Build ilass alignment engine (requires Rust + FFmpeg dev libs)
# macOS: brew install pkg-config ffmpeg
cd vendor/ilass && cargo build --release && cd ../..
```

### 2. Translate Videos

```bash
# Translate all MKV/MP4 files in a directory
uv run movie-translator ~/Downloads/anime

# Or with just:
just run ~/Downloads/anime
```

The tool will:
1. Find all `.mkv` and `.mp4` files (recursing one level of subdirectories)
2. Skip files that already have Polish subtitles
3. Search for Polish subtitles online, validate and align them
4. Fall back to AI translation if no suitable match is found
5. Mux Polish + English tracks into the video

### 3. Common Options

```bash
# Preview without modifying originals
uv run movie-translator ~/Downloads/anime --dry-run

# Disable online subtitle fetching (AI-only)
uv run movie-translator ~/Downloads/anime --no-fetch

# Adjust batch size for memory/speed tradeoff
uv run movie-translator ~/Downloads/anime --batch-size 8

# Use CPU instead of Apple Silicon GPU
uv run movie-translator ~/Downloads/anime --device cpu

# Show all options
uv run movie-translator --help
```

## How It Works

```
Video file (MKV/MP4)
  |
  +-> Identify media (title, season, episode, file hashes)
  +-> Extract English subtitle track (ASS/SRT/PGS via FFmpeg)
  |
  +-> Search subtitle providers (AnimeSub, Podnapisi, NapiProjekt, OpenSubtitles)
  +-> Download all Polish candidates
  +-> Validate candidates against English reference (line-level timing match)
  +-> Select best candidates (keep multiple if score >= 0.8)
  +-> Align to video timeline (ilass DP alignment / cross-correlation fallback)
  |
  +-> AI translation fallback (Allegro BiDi model on MPS/CPU)
  +-> Check if embedded fonts support Polish characters
  +-> Create subtitle tracks (fetched Polish + AI Polish + English)
  +-> Mux into video (Polish as default track)
```

### Dialogue Detection

ASS subtitle files often contain thousands of non-dialogue events (karaoke, signs, typesetting). The structural classifier identifies dialogue styles based on event properties rather than style name keywords:

- **Positioning ratio** -- signs and karaoke use explicit `\pos()`/`\move()`
- **Text length** -- karaoke syllables are 1-3 characters
- **Event density** -- karaoke has many rapid-fire short events

This approach works across arbitrary fansub naming conventions without maintaining a keyword list.

## Development

```bash
uv sync --group dev

just ci          # Run all checks and tests (CI equivalent)
just lint        # Auto-fix linting and formatting
just check       # Check without modifying (lint + format + type check)
just test        # Run tests
just run         # Run the CLI
```

### Project Structure

```
movie_translator/
  main.py                  # CLI entry point, file discovery
  pipeline.py              # 7-stage translation pipeline
  context.py               # Pipeline state (progressive population)
  types.py                 # Core types (DialogueLine, SubtitleFile)
  stages/                  # Pipeline stages
    identify.py            #   Media identification
    extract_ref.py         #   English subtitle extraction
    fetch.py               #   Online subtitle search + download + alignment
    extract_english.py     #   English source selection
    translate.py           #   AI translation
    create_tracks.py       #   Subtitle file creation
    mux.py                 #   Video muxing
  subtitle_fetch/          # Subtitle discovery and alignment
    fetcher.py             #   Multi-provider search
    validator.py           #   Timing-based validation
    style_classifier.py    #   Structural dialogue detection
    align.py               #   Cross-correlation alignment (built-in)
    align_ilass.py         #   ilass DP alignment (primary)
    scoring.py             #   Release name matching
    providers/             #   AnimeSub, Podnapisi, NapiProjekt, OpenSubtitles
  subtitles/               # Subtitle extraction and processing
  translation/             # AI translation (Allegro BiDi)
  inpainting/              # Burned-in subtitle removal
  video/                   # Video muxing and verification
  ocr/                     # Burned-in subtitle OCR (Apple Vision)
  identifier/              # Media identification (filename parsing, hashing)
scripts/
  gather_subtitle_data.py  # Corpus data collection for analysis
vendor/
  ilass/                   # Subtitle alignment engine (Rust, built from source)
```

## License

MIT License - see LICENSE file for details.
