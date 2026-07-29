# Project State

> Owner: parent orchestrator. Update after major milestones or architecture changes.
> Read protocol: parent reads relevant sections; children inspect repo directly.

## Architecture

English→Polish video subtitle translator. Pure-Rust Cargo workspace.
Crates: `mt-core`, `mt-subtitles`, `mt-discovery`, `mt-fetch`, `mt-media`, `mt-ml`, `mt-pipeline`, `mt-cli`.
Binary: `movie-translator` + `anime-dl`. Zero Python/PyO3.

ML: Apple Translation (macOS 26+), Apple Vision OCR (macOS), pure-Rust Telea inpainting.
Subtitle alignment: ilass DP (primary) + cross-correlation fallback.
Filename parsing: `anitomy-pure` + regex.

## Stable Invariants

- Pure Rust. No Python, PyO3, or venv.
- One serialized GPU worker (`worker.rs`). Every translate/OCR/inpaint awaits completion before next pull.
- Dialogue detection structural (position ratio, length, density). NOT keyword-based.
- Multiple viable subtitle candidates preserved per track.
- macOS-only Apple bridges (OCR, Translation). Linux errors gracefully.
- `rust-toolchain.toml` = single source of truth for compiler version.

## Important Commands

- Build: `just setup` (submodules + release build), `just brew` first on macOS
- Full gate: `just check && just test`
- Format: `just fix` (auto-format + clippy fix + Cargo.toml sort)
- Import check: `just check-imports`
- Run: `just run <file/dir> [flags]`
- Extract subs: `just extract <file/dir> [flags]`
- Anime download: `just anime-dl "<name>"`

## Repository Conventions

- Import hygiene: qualified path calls must have `use` import and short-form call.
- No unnecessary type annotations (exceptions: `collect()`, serde).
- After import changes: `cargo +nightly fmt` (sorts imports with `StdExternalCrate`).
- One writer per worktree. Max 3 concurrent children.
- Parallelize by disjoint file lanes (crates vs tooling/docs).

## Current Implementation Status

- Core pipeline: stable. Extraction, translation, muxing operational.
- Subtitle providers: AnimeSub, Podnapisi, NapiProjekt, OpenSubtitles with rate limiting.
- Alignment: ilass DP + cross-correlation fallback. Static offsets common (1-3s or 60-90s).
- Validation: timing-overlap score ≥ 0.8. Leading signs/karaoke skews scores — known weak spot.
- ML: Apple Translation + OCR macOS-only, Telea inpainting cross-platform.
- CLI: clap + ratatui TUI. `movie-translator` and `anime-dl` binaries.
