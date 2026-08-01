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

### Anime downloader

Download anime episodes from ogladajanime.pl (Polish hardsubs, best quality,
no translation) from a canonical v2 JSON episode list.

```bash
just anime-dl episodes.json
# or
just anime-dl episodes.json --ui dashboard --episodes 1,2,3
# or
just anime-dl --input episodes.json --out ~/Videos/anime
```

### Resolving episode URLs

ogladajanime.pl is the primary source of anime episodes with Polish hardsubs.
Player embed URLs are behind Cloudflare Turnstile and anti-debug, so they must
be resolved in a real browser via the Tampermonkey userscript:

1. Install [Tampermonkey](https://www.tampermonkey.net/) (Chrome/Firefox)
2. Install `scripts/ogladajanime_resolver.user.js` (v4) as a new userscript
3. Navigate to `ogladajanime.pl/anime/{series}` in your browser
4. Click one of the panel buttons:
   - **⏬ All N episodes** — walks every episode, resolves curated mirrors,
     downloads `anime-{slug}.json` (canonical v2)
   - **▶ This episode** — resolves a single episode, downloads
     `anime-{slug}-ep{N}.json`
5. Run `just anime-dl anime-{slug}.json` to download the episode files

**Canonical JSON format (v2):**

```json
{
  "schema_version": 2,
  "source_page": "https://ogladajanime.pl/anime/one-piece",
  "resolved_at": "2026-07-30T12:00:00Z",
  "title": "One Piece",
  "episodes": [
    {
      "episode": 1,
      "mirrors": [
        {
          "host": "cda",
          "quality": "1080p",
          "subtitle_group": "MioroSubs",
          "url": "https://cda.pl/video/..."
        },
        {
          "host": "rumble",
          "quality": "720p",
          "subtitle_group": null,
          "url": "https://rumble.com/..."
        }
      ]
    }
  ]
}
```

- `schema_version` must be `2`. v1 files (flat `urls`) are accepted with a
  warning and normalized; re-export with userscript v4 when possible. Legacy
  `resolved`/`embed_url` files are rejected with an actionable error.
- `episodes[].mirrors[]` is required — at least one `{host, quality,
  subtitle_group, url}` record per episode (matches userscript v4 field
  names).

### Flags

| Flag | Default | Description |
| --- | --- | --- |
| `--input, -i PATH` / positional `.json` | — | Canonical JSON episode list |
| `--out DIR` | `./<slug>` | Output directory |
| `--episodes N,N,...` | all | Episode filter |
| `--episode-concurrency N` | 4 | Max concurrent episode downloads |
| `--host-concurrency N` | 1 | Max concurrent downloads per host |
| `--ui MODE` | auto | `dashboard` (TTY) or `plain` (piped) |
| `-v` | off | Debug logging |
| `--resume` | off | Resume from manifest (skip Done, retry Failed) |
| `--retry-failed` | off | Re-run Failed episodes from manifest |
| `--validate-only` | off | Validate only; download nothing |
| `--no-validate` | off | Skip ffprobe validation |
| `--validate-force` | off | Revalidate despite cached verdict |
| `--min-size-mb F` | 1.0 | Minimum file size (MiB) |
| `--min-duration-secs F` | 1.0 | Minimum media duration (s) |
| `--require-audio` | off | Audio stream required (else warn-only) |
| `--ffprobe-timeout SECS` | 15 | Per-file ffprobe timeout |
| `--retry-attempts N` | 3 | Transient-failure retries per episode |
| `--cb-threshold N` | 3 | Systemic failures before host circuit breaker opens |
| `--cb-cooldown-secs SECS` | 60 | Circuit-breaker cooldown |
| `--clean-invalid` | off | Delete invalid files instead of quarantining |
| `--manifest PATH` | `<out>/<slug>.anime-manifest.json` | Manifest path override |
| `--ytdlp-extra-args ARGS` | — | Extra arguments to each yt-dlp call |

### Exit codes

| Code | Meaning |
| --- | --- |
| 0 | All episodes downloaded and validated |
| 1 | Fatal error (input, I/O, internal) |
| 2 | Usage error |
| 3 | Partial success (some failed) |
| 4 | All episodes failed |
| 130 | Cancelled (Ctrl+C) |

### Validation

Downloaded files are verified with `ffprobe`: extension allowlist
(`mkv/mp4/webm/flv/mov/avi`), min size `--min-size-mb`, min duration
`--min-duration-secs`, video stream required, audio warn-only unless
`--require-audio`. Placeholder dimensions/durations are rejected. If ffprobe
is missing, validation degrades to extension + size with a warning. Verdicts are cached in the manifest keyed by size+mtime; `--validate-force` re-probes.
Failed files are quarantined to `<out>/.quarantine/` (dotdir; `--clean-invalid` deletes
them instead).

### Manifest, resume, retry

Each run writes an atomic manifest at `<out>/<slug>.anime-manifest.json`
(input identity, episode states, ffprobe verdicts, per-episode attempt
history capped at 8, failure history). Host circuit-breaker state is
run-scoped only — it is never persisted in the manifest.

```bash
# Resume an interrupted run: skip done, re-queue failed/cancelled
just anime-dl episodes.json --resume

# Only retry episodes that failed last run
just anime-dl episodes.json --retry-failed

# Validate inputs + existing outputs without downloading
just anime-dl episodes.json --validate-only
```

Transient download failures retry with exponential backoff
(`2s × 2ⁿ + jitter`, capped 60 s, `--retry-attempts`). Each host has a
circuit breaker: after `--cb-threshold` systemic failures it is excluded for
`--cb-cooldown-secs` (URL-specific failures don't count).

### Host priority

Mirrors are tried in curated order: **cda first, rumble second**, then
sibnet / vk / mega / ok / dood / myvi / google / hqq / voe / mp4upload.
`vk` URLs are canonicalized to `vkvideo.ru`. Output naming:
`<out>/<slug>-E{NN}` (zero-padded).

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
