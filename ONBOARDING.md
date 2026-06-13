# ONBOARDING — what can you do / how do I run it

The skimmable capability index for a newcomer (human or AI agent). For the
denser tooling map, gotchas, and per-workflow how-tos, read [`AGENTS.md`](AGENTS.md).

## What this is

A local English→Polish video subtitle translator — **pure Rust, zero Python**.
Point it at an MKV/MP4 (or a directory) and it produces new files with as many
Polish subtitle tracks as it can find or make: identify the media, extract the
English track (text or OCR), search subtitle providers, validate + align
fetched Polish subs, fall back to AI translation, and mux everything back in.

**Architecture:** a Rust Cargo workspace (`crates/mt-*`). Translation uses the
Apple Translation framework on macOS (Swift bridge, compiled on demand). OCR
uses Apple Vision (Swift bridge). Inpainting uses a pure Rust Telea algorithm.
Everything is native — no CPython, no PyO3, no Python scripts.

## Quick start

```sh
just brew     # macOS only — system tools (just, ffmpeg, git-lfs, pkg-config)
just setup    # Submodules + release binary
just run ~/Downloads/anime --dry-run
```

Prerequisites: a Rust toolchain (`rustup` — version pinned in
`rust-toolchain.toml`). Burned-in-subtitle OCR needs macOS Apple Silicon. Linux
users: install the `Brewfile` equivalents, then `just setup`. Translation and
ASR transcription require macOS 26+ (Apple Translation + SpeechAnalyzer).

## Subagents

Tracked under `.pi/skills/` — invoke via the Skill tool or subagent delegation.

| Agent                            | When to use                                                                                        | When NOT                                             |
| -------------------------------- | -------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| **`gate-verify`**                | After edits: run the full gate (`just check` + `test`), report green or the exact failing gate.    | When you just want to make a fix.                    |
| **`ml-stage-debug`**             | A translation/OCR/inpainting stage misbehaves at runtime (all pure Rust — no Python).              | CLI/TUI bugs (use the main agent).                   |
| **`subtitle-fetch-align-debug`** | Fetched subs wrong/rejected/mis-timed: providers, validation, alignment, dialogue classification.  | Anything ML/GPU (use `ml-stage-debug`).              |
| **`benchmark-runner`**           | Audit stored benchmark results in git for regressions after a refactor. Read-only.                 | General correctness checks (use `gate-verify`).      |

## Just recipes

Run `just` (no args) for the live list. Key recipes:

| Recipe           | Purpose                                                                    | DESTRUCTIVE?                   |
| ---------------- | -------------------------------------------------------------------------- | ------------------------------ |
| `setup`          | Bootstrap a fresh checkout: submodules + release binary.                    |                                |
| `build`          | Release binary + vendored ilass alignment engine.                          |                                |
| `run <input>`    | Translate (default subcommand). `--dry-run` to preview without writing.    | with `--in-place` (overwrites) |
| `extract <input>`| Pull subtitles out (text + OCR), no translation.                           |                                |
| `check`          | clippy `-D warnings` + `cargo fmt --check` (mirrors CI).                   |                                |
| `test`           | Rust test suite (`cargo test --workspace`).                                |                                |
| `fix`            | Auto-format Rust, TOML, shell, Swift, JSON.                                |                                |
| `fix-clippy`     | Auto-fix clippy warnings.                                                  |                                |
| `tidy`           | `fix` + `fix-clippy` + dependency ordering check.                          |                                |
| `check`          | clippy + format checks for all file types (no modifications).              |                                |
| `ci`             | `check` + `test` (CI equivalent).                                          |                                |

## The gate chain

```
just check  →  clippy -D warnings + cargo fmt --check
just test   →  cargo test --workspace
just ci     =  check + test
```

Run `just check && just test` before claiming work done.


