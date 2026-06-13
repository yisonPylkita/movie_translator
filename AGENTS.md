# AGENTS.md — repo tooling map for AI agents (and new humans)

This doc is the one-shot orientation for working in this repo. If you're
an LLM coding agent (pi, Cursor, Aider, anything) or a new
contributor, read this first — it's a tighter, denser starting point
than the user-facing `README.md`.

## What this repo is

A local English→Polish video subtitle translator. Point it at a video
(MKV/MP4) or a directory and it produces new files with as many Polish
subtitle tracks as it can find or make: it identifies the media from the
filename, extracts the English track (text or OCR for burned-in/PGS),
searches subtitle providers for existing Polish subs, validates and
aligns them to the video's timeline, falls back to AI translation when
nothing is found, and muxes everything back in.

**Architecture (the load-bearing fact):** everything is **pure Rust** in a
**Cargo workspace** (`crates/mt-*`) built into the `movie-translator`
binary. There is **zero Python dependency** — no CPython embedding, no
PyO3, no venv, no Python scripts. Translation uses the Apple Translation
framework on macOS (Swift bridge via subprocess, compiled on demand).
OCR uses Apple Vision (Swift bridge). Inpainting uses a pure Rust Telea
algorithm. Filename parsing uses `anitomy-pure` + regex. All ML inference
is native. Design history: `docs/superpowers/specs/2026-05-27-rust-rewrite-design.md`.

## How to do common things

Each is one paragraph, terse on purpose. Where a subagent fits the
workflow, its name is in **bold**.

### Run the full gate (before any "done")

`just check && just test`. `check` = `cargo clippy --workspace
--all-targets -D warnings` + `cargo +nightly fmt --check` + `just check-imports`
(ast-grep import hygiene). `test` = `cargo test --workspace`. `just ci`
runs `check + test`. Cite the output; never assert green without it.
Subagent: **`gate-verify`**.

### Translate / extract a video

`just run <file-or-dir> [flags]` (translate; default subcommand) or
`just extract <file-or-dir> [flags]` (pull subtitles out, no
translation). Both wrap `cargo run --release`. Key flags: `--dry-run`
(no writes), `--no-fetch` (AI only), `--model apple` (only the Apple
Translation backend is supported; macOS 26+ required), `--workers N
--batch-size N`, `--inpaint` (remove burned-in subs — slow),
`--in-place` (overwrite originals — destructive, NOT compatible with
`--inpaint`), `--keep-artifacts` (leave intermediates in
`.translate_temp/`), `--hardsub-ocr` (source a Polish
track by OCRing burned-in subs from ogladajanime.pl — opens the browser, you
run `scripts/ogladajanime_resolver.user.js`, it picks up the JSON from
`~/Downloads`; macOS-only), `--force` (re-process files that already have
Polish), `--transcribe [--transcribe-engine {apple,whisper}]` (source English
dialogue from the audio track via ASR when no subtitle text is found — Apple
SpeechAnalyzer on macOS 26+ or mlx-whisper large-v3; bake-off in
`benchmarks/asr/REPORT.md`). Real runs need GPU + model files; for iteration
prefer `--dry-run` or a tiny synthetic clip.

### Download an anime season (no translation)

`just anime-dl "<anime name>" [--out DIR] [--timeout SECS] [--json PATH]`
(wraps `cargo run --release --bin anime-dl`). Standalone binary, separate from
`movie-translator`: it finds the anime on ogladajanime.pl, opens the browser,
waits for the resolver userscript to drop its players JSON in `~/Downloads`,
then downloads **every** episode at **best** available quality (no translation,
no OCR). Reuses `mt_fetch::ogladajanime` (discover/open/poll/parse) and
`mt_ml::hardsub_download(..., best=true, ...)`. `--json` skips the browser and
parses an existing resolver JSON (cheap re-runs). Sequential, mirror-fallback
per episode; resumes by skipping episodes already on disk. macOS-oriented
(browser + userscript flow); the `--hardsub-ocr` translate flag is the
OCR-into-the-pipeline cousin.

### Build / set up a fresh checkout

`just setup` (submodules + build; idempotent). On macOS run `just brew`
once first. The pieces: `just submodules` (vendored `ilass`), `just
build` (release binary + vendored ilass). No Python or venv needed.

### Debug an ML stage (translation / OCR / inpainting)

All ML stages run in pure Rust. Translation calls the Apple Translation
framework on macOS via a compiled Swift bridge subprocess. OCR calls
Apple Vision via Swift bridge. Inpainting uses a pure Rust Telea
algorithm. Filename parsing uses `anitomy-pure` + regex. If a stage
fails, check the Swift bridge compilation output or the ffmpeg subprocess
stderr. There is no Python layer to debug.

### Debug an ML stage (translation / OCR / inpainting)

All ML stages are pure Rust. Translation = `crates/mt-ml/src/translate/apple.rs`
(Apple Translation via Swift bridge + Rust sentence merger).
OCR = `crates/mt-ml/src/ocr/vision.rs` (Apple Vision via Swift bridge).
Inpainting = `crates/mt-ml/src/inpaint.rs` (pure Rust Telea algorithm).
Subagent: **`ml-stage-debug`**.

### Debug subtitle fetch / validation / alignment

Fetched subs are timed to a different release than the user's video, or
a provider returns junk, or validation rejects a good file. Fetch +
scoring + alignment all live in `crates/mt-fetch` (`fetcher.rs`,
`providers/`, `validator.rs`, `scoring.rs`, `align.rs`,
`align_ilass.rs`, `style_classifier.rs`). Validation scores fetched
subs against the English reference by timing overlap; candidates ≥ 0.8
are kept (multiple tracks allowed). Alignment is ilass (DP) with a
cross-correlation fallback. Subagent: **`subtitle-fetch-align-debug`**.

### Run / read benchmarks

ASR benchmarks and translation benchmarks live in `benchmarks/`.
Convention: after a big refactor, run the benchmark and commit the
results into git so quality regressions are visible in history.
Subagent: **`benchmark-runner`**.

### Auto-fix format + lint

`just fix` formats all files, auto-fixes clippy warnings, and sorts Cargo.toml
dependencies. `just check` validates everything without modifying (includes
clippy, rustfmt, and import hygiene via ast-grep).

### Check import hygiene

`just check-imports` runs `sg scan` using the pi-lens ast-grep rule at
`.pi/rules/ast-grep-rules/rules/import-function-over-path.yml`. Reports
fully-qualified calls that should use `use` imports instead.
Requires `sg` (ast-grep) to be installed.

## File map

```
.
├── crates/                         Rust Cargo workspace (the binary)
│   ├── mt-core/        Foundation: error, types, context, exec, identity. No mt-* deps.
│   ├── mt-subtitles/   ASS/SRT parsing, encoding, dialogue model, processor.
│   ├── mt-discovery/   Filename → media identity (anitomy-pure + regex), hashing, TMDB.
│   ├── mt-fetch/       Providers, download, validate, score, align (ilass + xcorr), style classifier.
│   ├── mt-media/       FFmpeg extract/mux, font checks, file operations, PGS parser.
│   ├── mt-ml/          ML inference: Apple Translation (Swift bridge), Apple Vision OCR, inpainting.
│   ├── mt-pipeline/    Orchestration: stages, GPU worker (serialised), progress events, proper nouns.
│   └── mt-cli/         clap CLI + ratatui TUI. Two bins: movie-translator + anime-dl (season downloader).
├── vendor/ilass/                   Git submodule — DP subtitle alignment engine (built by `just build`).
├── benchmarks/                     ASR + translation benchmarks.
├── docs/
│   ├── research/                   Custom-translation-model design (research only).
│   └── superpowers/{plans,specs}/  Historical design docs; rust-rewrite spec is the architecture record.
├── justfile                        Tooling entry point (`just` to list).
├── Cargo.toml / Cargo.lock         Rust workspace.
├── rustfmt.toml                    Rustfmt configuration.
├── rust-toolchain.toml             Pinned Rust channel (single source of truth, local + CI).
├── Brewfile                        macOS system tools.
├── README.md                       User-facing docs.
├── AGENTS.md                       This file.
├── .pi/                            Pi-lens skills and agent definitions.
├── scripts/                        Utility scripts.
```

## Gotchas list

Every entry below cost real debugging time. Read once, save hours later.

### GPU / pipeline serialization

- **One GPU, one worker.** Every translate/OCR/inpaint goes through a
  single tokio task (`worker.rs`) that `await`s each job to completion
  before pulling the next — that's the serialization guarantee. The
  synchronous `GpuExecutor` trait methods block on a oneshot reply and
  are only safe because they're always called from `spawn_blocking`
  threads, never a runtime worker thread. Don't call them off a runtime
  thread; don't parallelize GPU calls.
- **Files parallelize; GPU stages don't.** Discovery, fetch, validation,
  mux run concurrently across files. Only the GPU stages funnel through
  the one worker. Keep that boundary when adding stages.

### Subtitles / fetch / alignment (`crates/mt-fetch`, `mt-subtitles`)

- **Dialogue detection is structural, not keyword-based.** ASS files
  carry thousands of non-dialogue events (karaoke, signs, typesetting).
  The classifier (`style_classifier.rs`) uses positioning ratio, text
  length, and event density — NOT style-name keywords — so it works
  across arbitrary fansub naming. Don't "fix" it by adding a keyword
  list.
- **Validation can mis-handle files whose non-dialogue events precede
  the first dialogue line.** Known weak spot — see
  `[[project_validation_bug]]`. If a clearly-good fetched sub scores
  low, check whether leading signs/karaoke events are skewing the
  timing-overlap score.
- **Fetched subs often need a static offset.** Different encode start
  points (1-3 s) or OP/ED removal (60-90 s+) shift the whole file.
  Alignment handles small/large/piecewise offsets; a known case is a
  constant offset on Polish subs (observed on Konosuba S1E1). Primary =
  ilass DP, fallback = cross-correlation.
- **Keep multiple candidates.** Validation keeps every candidate scoring
  ≥ 0.8 as a separate output track — "best one wins" is wrong here; the
  user wants all viable Polish tracks.
- **Providers are rate-limited and external.** AnimeSub, Podnapisi,
  NapiProjekt, OpenSubtitles. There's a `rate_limiter.rs` + `retry.rs`
  for a reason — don't loop fetches in a tight retry without backoff.

### ML stages (`crates/mt-ml`)

- **OCR is macOS-only** (Apple Vision via Swift bridge). On Linux,
  burned-in OCR and PGS OCR return errors gracefully.
- **Apple Translation builds a Swift bridge** from source on first use
  (`movie_translator/translation/swift/translate_bridge`). The binary is
  gitignored and compiled on demand. macOS 26+ required.
- **Inpainting uses a pure Rust Telea algorithm** — no OpenCV or Python
  dependency. Works on any platform.

### Build / toolchain / CI

- **`rust-toolchain.toml` is the single source of truth** for the
  compiler version (local + CI both read it via `rustup show`). Bump
  deliberately.
- **`.translate_temp/` and `test_workdir/` are scratch** (gitignored).
  `--keep-artifacts` populates the former for debugging.

### Import hygiene rules (MUST follow every edit)

Every function, macro, or type used with a qualified path (`std::process::exit`,
`tracing::info!`, `serde_json::from_str`, `tempfile::tempdir`, `tokio::spawn`, etc.)
must be imported at the top of the file with `use` and called in short form.

**Correct:**
```rust
use std::process::exit;
use tracing::{info, warn, error};
use serde_json::{from_str, Value};

info!("translation complete");
exit(0);
let val: Value = from_str(json)?;
```

**Incorrect:**
```rust
tracing::info!("translation complete");
std::process::exit(0);
let val: serde_json::Value = serde_json::from_str(json)?;
```

**Exceptions** (keep fully-qualified):
- `#[from] std::io::Error` / `#[source] std::io::Error` in derive macros
- `pub type Result<T> = std::result::Result<T, MtError>;` (type alias)
- Doc-comment backtick references like ``[`tokio::sync::Semaphore`]``
- `clap::Parser` / `thiserror::Error` / `serde::Deserialize` in derives
  (these are imported and used in derives, which is fine)

**After adding ANY import, ALWAYS run `cargo +nightly fmt`.**
rustfmt sorts imports into groups (std → external → crate) and
alphabetically within groups. Skipping this step will produce
format-check failures at the gate.

**Import hygiene is checked automatically** by `just check` via the ast-grep
rule at `.pi/rules/ast-grep-rules/rules/import-function-over-path.yml`. Run
`just check-imports` directly to see all violations across the workspace.
Note: the rule currently flags many one-off calls (`Vec::new()`,
`Path::new()`) that are idiomatic — it will be refined to only flag
repeat offenders.

### No unnecessary type annotations

Don't write `let x: Vec<String> = Vec::new()` — the RHS makes the type
obvious. Remove the annotation:

```rust
// Bad
let mut rebuilt: Vec<String> = Vec::with_capacity(argv.len());

// Good
let mut rebuilt = Vec::with_capacity(argv.len());
```

**Exceptions** (annotation needed):
- `let x: HashSet<_> = expr.collect();` — collect() needs a collection hint
- `let x: SomeType = serde_json::from_str(json)?;` — serde needs the target type
  (or use turbofish: `from_str::<SomeType>(json)?`)

### Agentic workflow rules

- **Parallelize by DISJOINT file lanes.** Two non-overlapping lanes:
  Rust crates (`crates/**`) and tooling/docs (`justfile`, `.github/**`,
  `docs/**`). Fan out one agent per lane; they never touch each other's
  files. Same-file work is serial.
- **Serialize GPU + outward actions.** GPU is a single shared resource
  (see above). `--in-place`, `git lfs` ops, and looped provider fetches
  are destructive/outward — confirm, don't parallelize.
- **Verify via the gate chain, cite evidence.** Run `just check` / `just test`
  and quote the output; never assert "done" without having run it.
- **After any batch of edits that touches imports, run `cargo +nightly fmt` before
  claiming done.** This fixes import ordering automatically and avoids
  the most common gate failure.
- **No code index — plain grep/ripgrep is the default.** For structural Rust
  edits across crates, prefer compiler-driven refactors (rename via the type
  system, then `just check`) over text munging.

## Subagent index

Tracked under `.pi/skills/`. Each is a pi skill definition — invoke via
the Skill tool or subagent delegation. Reach for these when the task
spans several crates/tools and has multiple gotchas; for one-shot file
edits or simple queries, use the main agent directly.

| Subagent                       | When to use                                                                                                                                                  |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **`gate-verify`**              | After edits, want to ship-check: runs `just check` + `test`, parses each, reports green or the exact failing gate/test with output. Read-only.    |
| **`ml-stage-debug`**           | A translation / OCR / inpainting stage misbehaves at runtime. Debugs the Rust/Swift bridge code path.    |
| **`subtitle-fetch-align-debug`** | Fetched subs are wrong/rejected/mis-timed: provider issues, validation scoring, ilass/xcorr alignment, dialogue classification, offset correction.          |
| **`benchmark-runner`**         | Run the translation-quality benchmarks and/or audit stored results in git for regressions after a refactor.                                                  |
