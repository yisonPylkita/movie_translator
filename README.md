# Movie Translator

[![Tests](https://github.com/yisonPylkita/movie_translator/actions/workflows/tests.yml/badge.svg)](https://github.com/yisonPylkita/movie_translator/actions/workflows/tests.yml)

**AI subtitle translator** for English to Polish translation. Fetches existing
Polish subtitles from the internet when available, falls back to AI translation,
and handles timing alignment automatically. Runs entirely locally on your
machine — **pure Rust, zero Python**.

> **Architecture.** Everything is a Rust Cargo workspace (`crates/mt-*`) built
> into the `movie-translator` binary. Translation uses the Apple Translation
> framework on macOS (Swift bridge, compiled on demand). OCR uses Apple Vision
> (Swift bridge). Inpainting uses a pure Rust Telea algorithm. Filename parsing
> uses `anitomy-pure` + regex. No CPython, no PyO3, no Python scripts.
> Design history: `docs/superpowers/specs/2026-05-27-rust-rewrite-design.md`.

## What It Does

Takes video files (MKV or MP4) and produces new video files with as many Polish
subtitle tracks as possible. The pipeline is fully automated:

1. **Identifies** the media (title, season, episode) from the filename
2. **Extracts** the English subtitle track from the video (text or OCR for
   burned-in/PGS)
3. **Searches** multiple subtitle providers for existing Polish translations
4. **Validates** downloaded subtitles against the English reference (timing-based
   scoring)
5. **Aligns** fetched subtitles to the video's timeline (handles OP removal,
   different video cuts)
6. **Translates** with AI as a fallback when no Polish subtitles are found
   online
7. **Muxes** everything back into the video — all Polish tracks plus the
   original English

There is also a standalone **extract** command for pulling subtitles out of
videos (text tracks and burned-in OCR) without translating.

### Subtitle Sources

All available sources are included as separate tracks in the output:

- **Internet downloads** from AnimeSub, Podnapisi, NapiProjekt, and
  OpenSubtitles
- **AI translation** using Apple Translation (macOS 26+)
- **External pre-extracted subtitles** via `--external-subs` (from a prior
  `extract` run)

### Subtitle Alignment

Fetched subtitles are often timed to different video releases. The alignment
system handles:

- **Small offsets** (1-3s) from different video encode start points
- **Large offsets** (60-90s+) from OP/ED removal in the subtitle source video
- **Piecewise alignment** with different offsets for pre-OP and post-OP segments

Primary alignment uses [ilass](https://github.com/SandroHc/ilass) (DP algorithm
with split penalties), with a built-in cross-correlation fallback.

## Quick Start

Everything is driven by [`just`](https://github.com/casey/just). Run `just` (or
`just --list`) at any time to see every available recipe.

```bash
git clone --recurse-submodules https://github.com/yisonPylkita/movie_translator.git
cd movie_translator

# macOS only — installs system tools: just, ffmpeg, git-lfs, pkg-config.
just brew

# One-shot setup: submodules + release binary.
just setup

# Translate an episode (dry-run first to preview without writing):
just run ~/Downloads/episode.mkv --dry-run
# Remove --dry-run to actually write output.
```

Prerequisites: a Rust toolchain (`rustup` — version pinned in
`rust-toolchain.toml`). **Translation and ASR transcription require macOS 26+**
(Apple Translation + SpeechAnalyzer). Burned-in-subtitle OCR needs macOS Apple
Silicon (Apple Vision). Linux users: install the Brewfile equivalents, then `just
setup`; the pipeline works but translation falls back gracefully when Apple
frameworks are unavailable.

## Key Recipes

| Recipe             | Purpose                                                                 |
| ------------------ | ----------------------------------------------------------------------- |
| `just setup`       | First-time setup: submodules + release binary                          |
| `just build`       | Release binary + vendored ilass                                        |
| `just run <input>` | Translate (default). `--dry-run` to preview without writing            |
| `just extract <input>` | Pull subtitles out (text + OCR), no translation                    |
| `just test`        | Rust test suite (`cargo test --workspace`)                              |
| `just check`       | Clippy `-D warnings` + `cargo fmt --check` (mirrors CI)                |
| `just fix`         | Format all, auto-fix clippy, sort Cargo.toml dependencies                |
| `just ci`          | `check` + `test` (CI equivalent)                                        |

## Gate Chain

```sh
just check   → clippy -D warnings + cargo fmt --check
just test    → cargo test --workspace
just ci      = check + test
```

Run `just check && just test` before committing.

## Advanced Usage

### Flags

- `--dry-run` — preview what would happen without writing any output files.
- `--inpaint` — remove burned-in subtitles via inpainting (slow, macOS only).
- `--in-place` — overwrite the original file instead of creating a new one
  (NOT compatible with `--inpaint`).
- `--keep-artifacts` — leave intermediate files in `.translate_temp/` for
  debugging.
- `--hardsub-ocr` — source a Polish track by OCRing burned-in subs from
  ogladajanime.pl.
- `--transcribe` — source English dialogue from the audio track via ASR when no
  subtitle text is available.
- `--model apple` — use the Apple Translation backend (macOS 26+, the only
  supported backend).

### Anime Downloader

Download whole anime seasons for offline OCR processing:

```bash
just anime-dl "One Piece"
```

This opens ogladajanime.pl in the browser, waits for the resolver userscript,
then downloads every episode at best available quality (no translation, no OCR).

## macOS Dependencies

The following are required on macOS (installed via `just brew`):

1. **just** — command runner (the entry point for everything)
2. **FFmpeg** — media processing (frame extraction, audio extraction, muxing)
3. **git-lfs** — large file storage
4. **pkg-config** — system library discovery
5. **Xcode Command Line Tools** — for `swiftc` (compiles the Apple Translation,
   Vision OCR, and SpeechAnalyzer bridges on first use)

**Translation** and **ASR transcription** need macOS 26+. OCR needs macOS
(Apple Silicon recommended).

## Linux

Install the Brewfile equivalents from your distro's package manager, then
`just setup`. The pipeline works, but translation/OCR features that depend on
Apple frameworks are unavailable and gracefully return `None`.

## Docs

- [`AGENTS.md`](AGENTS.md) — tooling map, subagent index, gotchas list
- [`ONBOARDING.md`](ONBOARDING.md) — capability index, quick start, subagent
  table
- [`docs/superpowers/`](docs/superpowers/) — design history and specs
