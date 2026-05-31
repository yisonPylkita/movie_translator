# ONBOARDING — what can you do / how do I run it

The skimmable capability index for a newcomer (human or AI agent). For the
denser tooling map, gotchas, and per-workflow how-tos, read [`AGENTS.md`](AGENTS.md).

## What this is

A local English→Polish video subtitle translator. Point it at an MKV/MP4 (or a
directory) and it produces new files with as many Polish subtitle tracks as it
can find or make: identify the media, extract the English track (text or OCR),
search subtitle providers, validate + align fetched Polish subs, fall back to AI
translation, and mux everything back in.

**Architecture:** a Rust Cargo workspace (`crates/mt-*`) is the CLI + pipeline;
the machine learning (translation / OCR / inpainting) and filename parsing are a
Python package (`movie_translator/`) that the Rust binary **embeds via PyO3**
(no subprocess). That split is the single most important fact about this repo.

## Quick start

```sh
just brew     # macOS only — system tools (just, ffmpeg, git-lfs, uv, pkg-config)
just setup    # Python venv + submodules + git-lfs model + release binary
just run ~/Downloads/anime --dry-run
```

Prerequisites: a Rust toolchain (`rustup` — version pinned in
`rust-toolchain.toml`). Burned-in-subtitle OCR needs macOS Apple Silicon (Apple
Vision). Linux: install the `Brewfile` equivalents from your package manager,
then `just setup`.

## Subagents

Tracked under [`.claude/agents/`](.claude/agents/) — invoke via the Agent tool.
Reach for these when a task spans several crates/tools and has multiple gotchas;
for one-shot edits or simple queries use the main agent directly.

| Agent                            | When to use                                                                                                                | When NOT                                                                              |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **`gate-verify`**                | After edits: run the full gate chain (`just check` + `test` + `py-test`), report green or the exact failing gate. Read-only. | When you just want to make a fix — verify after, not instead.                         |
| **`pyo3-bridge-doctor`**         | The Rust↔Python boundary is broken: import failures, `PYO3_PYTHON` mismatch, libpython/`LD_LIBRARY_PATH`, mp-spawn crashes. | Pure Python ML logic bugs (use `ml-stage-debug`); pure Rust logic.                    |
| **`ml-stage-debug`**             | A translation/OCR/inpainting stage misbehaves at runtime; split ML bug from bridge bug and localize the module.            | The bridge itself failing to import (use `pyo3-bridge-doctor`).                        |
| **`subtitle-fetch-align-debug`** | Fetched subs wrong/rejected/mis-timed: providers, validation scoring, ilass/xcorr alignment, dialogue classification.      | Anything ML/GPU (use `ml-stage-debug`); CLI/TUI bugs.                                  |
| **`benchmark-runner`**           | Run the translation-quality benchmarks, or audit stored results in git for regressions after a refactor.                   | General correctness checks (use `gate-verify`).                                        |

## Just recipes

Run `just` (no args) for the live list. Key recipes:

| Recipe              | Purpose                                                                                  | DESTRUCTIVE?                  |
| ------------------- | ---------------------------------------------------------------------------------------- | ----------------------------- |
| `setup`             | Bootstrap a fresh checkout: deps + submodules + git-lfs model + release binary.          |                               |
| `build`             | Release binary + vendored ilass alignment engine.                                        |                               |
| `run <input>`       | Translate (default subcommand). `--dry-run` to preview without writing.                  | with `--in-place` (overwrites)|
| `extract <input>`   | Pull subtitles out (text + OCR), no translation.                                         |                               |
| `check`             | clippy `-D warnings` + `cargo fmt --check` + ruff (no modifications — mirrors CI).       |                               |
| `test`              | Rust test suite (`cargo test --workspace`).                                              |                               |
| `py-test`           | Python ML-backend test suite (pytest).                                                   |                               |
| `lint`              | Auto-fix lint + format (Rust + Python).                                                   |                               |
| `ci`                | `check` + `test` (the Rust half of CI).                                                  |                               |
| `clean`             | Remove Rust build artifacts.                                                             |                               |

## The gate chain

```
just check     →  clippy -D warnings + cargo fmt --check + ruff       (no compile of tests)
  → just test  →  cargo test --workspace                              (Rust suite)
  → just py-test  →  pytest over the movie_translator package          (Python ML suite)
```

`just ci` = `check + test`. **`py-test` is run separately** — run all three
before claiming work done. CI runs all of it on Linux + macOS; it deliberately
skips the git-lfs model (real-model tests are `#[ignore]`'d).

## Reusable workflows

Named multi-agent workflows live under [`.claude/workflows/`](.claude/workflows/)
and are invoked via the Workflow tool. Two are shipped, both read-only:

| Workflow             | What it does                                                                                                              |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **`review-changes`** | Read-only 3-lane parallel review of the working-tree diff (Rust crates / Python ML / tooling+docs), then a prioritized synthesis. |
| **`benchmark-audit`** | Read-only audit of stored benchmark results in git — current scores, delta vs last run, flags regressions and staleness. |

See [Agentic workflow rules](AGENTS.md) for the lane-partition and
serialize-GPU conventions they follow.

## Skills

This environment exposes several skill families, invoked via the Skill tool (or
the matching `/slash-command`). The **superpowers** family covers process
discipline — `brainstorming`, `systematic-debugging`, `test-driven-development`,
`dispatching-parallel-agents`, plus planning/verification/code-review variants.
There are also standalone skills such as `deep-research`, `code-review`,
`security-review`, `verify`, and `run`. Match the skill to the task and invoke
it before starting the work it governs (e.g. `brainstorming` before new
features, `systematic-debugging` before fixing a bug).
