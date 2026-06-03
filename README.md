# Movie Translator

[![Tests](https://github.com/yisonPylkita/movie_translator/actions/workflows/tests.yml/badge.svg)](https://github.com/yisonPylkita/movie_translator/actions/workflows/tests.yml)

**AI subtitle translator** for English to Polish translation. Fetches existing Polish subtitles from the internet when available, falls back to AI translation, and handles timing alignment automatically. Runs entirely locally on your machine.

> **Architecture.** The CLI and the orchestration pipeline — subtitle parsing, fetching, validation, alignment, media/mux, filename discovery, GPU-worker serialization — are a Rust Cargo workspace (`crates/mt-*`), built into the `movie-translator` binary. Machine-learning inference (translation, OCR, inpainting) and `guessit`/`aniparse` filename parsing run as single-purpose Python scripts under `ml/`, backed by the importable `movie_translator` package as the ML backend; the Rust binary spawns them as needed. Design notes (history): `docs/superpowers/specs/2026-05-27-rust-rewrite-design.md`.

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

## Quick Start

Everything is driven by [`just`](https://github.com/casey/just). Run `just` (or `just --list`) at any time to see every available recipe.

```bash
git clone --recurse-submodules https://github.com/yisonPylkita/movie_translator.git
cd movie_translator

# macOS only — installs system tools: just, ffmpeg, git-lfs, uv, pkg-config.
just brew

# One-shot setup: Python env, submodules, model files, release binary.
just setup

# Translate a video file or a directory of videos.
just run ~/Downloads/anime
```

Linux: install `just`, `ffmpeg`, `git-lfs`, `uv`, and `pkg-config` from your distro's package manager (the equivalents of the `Brewfile`), then run `just setup`.

**Prerequisite for both:** a working [Rust toolchain](https://rustup.rs/) (`rustup`) — the pinned compiler version is read from `rust-toolchain.toml`. For burned-in subtitle OCR, macOS with Apple Silicon is required (uses Apple Vision).

> Already cloned without `--recurse-submodules`? `just setup` runs `git submodule update --init --recursive` for you.

## Usage

### Translate

```bash
# Translate every MKV/MP4 under a directory (or a single file)
just run ~/Downloads/anime

# Preview without modifying originals
just run ~/Downloads/anime --dry-run

# Disable online subtitle fetching (AI translation only)
just run ~/Downloads/anime --no-fetch

# Pick a translation backend (default on macOS runs both allegro AND apple)
just run ~/Downloads/anime --model apple

# Concurrency / batching / device
just run ~/Downloads/anime --workers 4 --batch-size 8 --device cpu

# Remove burned-in subtitles from video frames via inpainting (slow)
just run ~/Downloads/anime --inpaint

# Disk-frugal mode: replace originals in place (not compatible with --inpaint)
just run ~/Downloads/anime --in-place

# Keep intermediate artifacts under .translate_temp for debugging
just run ~/Downloads/anime --keep-artifacts

# All options
just run --help
```

The translate flow finds all `.mkv`/`.mp4` files recursively, skips files that already have Polish subtitles, searches and validates online candidates, falls back to AI translation when needed, and muxes every Polish track + the original English back into the video.

### Extract

Extract subtitles from videos without translating — useful for pulling burned-in Polish subtitles from a low-quality source to apply to a better version:

```bash
# Extract burned-in Polish subtitles via OCR
just extract ~/Downloads/polish_version --ocr-language pl

# Use the extracted SRTs when translating the high-quality version
just run ~/Downloads/english_version --external-subs ~/Downloads/polish_version/extracted_subs
```

The extract command outputs SRT files and a `manifest.json` that the translate command uses for matching by media identity (title + season + episode).

### Download a season (`anime-dl`)

Download a whole Polish-subbed anime season straight from ogladajanime.pl — no translation, just grab the episodes:

```bash
# Find the anime, open the browser, download every episode at best quality
just anime-dl "Isekai Ojisan"

# Custom output dir / longer wait for the userscript JSON
just anime-dl "Isekai Ojisan" --out ~/Anime/isekai-ojisan --timeout 900

# Skip the browser and reuse a resolver JSON you already have
just anime-dl --json ~/Downloads/oga-isekai-ojisan-all.players.json
```

`anime-dl` opens the matched anime page in your browser; you run the resolver userscript (`scripts/ogladajanime_resolver.user.js`), which enumerates the whole season and downloads a players JSON to `~/Downloads`. `anime-dl` picks that up and downloads each episode (best mirror first, falling back on dead ones), skipping episodes already on disk. macOS-oriented (browser + userscript flow).

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

Every common task is a `just` recipe — run `just --list` to see them all.

| Recipe | What it does |
| --- | --- |
| `just setup` | First-time setup: Python env, submodules, model files, release binary |
| `just build` | Rebuild the release binary + the vendored ilass engine |
| `just run <input>` | Translate (default subcommand) |
| `just extract <input>` | Extract subtitles only (no translation) |
| `just anime-dl "<name>"` | Download a whole anime season from ogladajanime.pl |
| `just test` | Run the Rust test suite |
| `just py-test` | Run the Python ML-backend test suite |
| `just check` | Clippy + fmt check + ruff (no modifications — mirrors CI) |
| `just lint` | Auto-fix lint + format (Rust + Python) |
| `just ci` | `check` + `test` (CI equivalent) |
| `just clean` | Remove Rust build artifacts |
| `just install-hooks` | Install a git pre-commit hook that runs `just check` |

## License

MIT — see `LICENSE`.
