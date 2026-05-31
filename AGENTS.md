# AGENTS.md — repo tooling map for AI agents (and new humans)

This doc is the one-shot orientation for working in this repo. If you're
an LLM coding agent (Claude Code, Cursor, Aider, anything) or a new
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

**Architecture (this is the load-bearing fact):** the CLI and the
orchestration pipeline are a **Rust Cargo workspace** (`crates/mt-*`)
built into the `movie-translator` binary. **Machine learning** —
translation, OCR, inpainting — and `guessit`/`aniparse` filename parsing
run as **Python** in the importable `movie_translator` package. The Rust
binary does NOT spawn `python *.py` subprocesses anymore: `crates/mt-ml`
**embeds CPython via PyO3** and calls the package directly. Design notes:
`docs/superpowers/specs/2026-05-27-rust-rewrite-design.md`.

## How to do common things

Each is one paragraph, terse on purpose. Where a subagent fits the
workflow, its name is in **bold**.

### Run the full gate (before any "done")

`just check && just test && just py-test`. `check` = `cargo clippy
--workspace --all-targets -D warnings` + `cargo fmt --check` + `ruff
check movie_translator/`. `test` = `cargo test --workspace`. `py-test` =
`pytest` over the `movie_translator` package. `just ci` runs `check +
test` (the Rust half). Cite the output; never assert green without it.
Subagent: **`gate-verify`**.

### Translate / extract a video

`just run <file-or-dir> [flags]` (translate; default subcommand) or
`just extract <file-or-dir> [flags]` (pull subtitles out, no
translation). Both wrap `cargo run --release`. Key flags: `--dry-run`
(no writes), `--no-fetch` (AI only), `--model {allegro,apple}`,
`--workers N --batch-size N --device {cpu,mps,cuda}`, `--inpaint`
(remove burned-in subs — slow), `--in-place` (overwrite originals —
destructive, NOT compatible with `--inpaint`), `--keep-artifacts`
(leave intermediates in `.translate_temp/`). Real runs need GPU + model
files; for iteration prefer `--dry-run` or a tiny synthetic clip.

### Build / set up a fresh checkout

`just setup` (deps + submodules + model + build; idempotent). On macOS
run `just brew` once first. The pieces: `just deps` (`uv sync` →
`.venv/`), `just submodules` (vendored `ilass`), `just model` (`git lfs
pull` the Allegro BiDi model), `just build` (release binary + vendored
ilass). The binary links libpython, so `deps` must precede `build`.

### Fix the Rust↔Python embedded-CPython boundary

`import movie_translator` failing, torch/transformers not loading,
multiprocessing workers crashing, "library not found" on Linux — these
are PyO3-embedding problems, not logic bugs. The contract: `PYO3_PYTHON`
= `.venv/bin/python` at build time; on Linux `LD_LIBRARY_PATH` must
include libpython's dir at runtime; `multiprocessing.set_executable` is
pinned to the venv python at interpreter init. Subagent:
**`pyo3-bridge-doctor`**.

### Debug an ML stage (translation / OCR / inpainting)

The stage runs inside the GPU worker via `crates/mt-ml` → the
`movie_translator` Python package. Reproduce the Python side in
isolation first (`uv run python -c "from movie_translator.translation
import ..."`) to split "Python ML bug" from "Rust bridge/pipeline bug."
Translation = `movie_translator/translation/` (Allegro model + Apple
backend + sentence merger). OCR = `movie_translator/ocr/` (Vision,
PGS, burned-in, frame extraction). Inpainting =
`movie_translator/inpainting/` (LaMa). Subagent: **`ml-stage-debug`**.

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

Translation-quality benchmarks live under `benchmarks/` (`uv sync
--group benchmarks` for the sacrebleu scorer). Convention: after a big
refactor, run the benchmark and commit the results into git so quality
regressions are visible in history. Subagent: **`benchmark-runner`**.

### Auto-fix lint / format

`just lint` (clippy `--fix` + `cargo fmt` + `ruff check --fix` + `ruff
format`). The PostToolUse hook already formats each file you edit; `just
lint` is the whole-tree version.

## File map

```
.
├── crates/                         Rust Cargo workspace (the binary)
│   ├── mt-core/        Foundation: error, types, context, exec, identity. No mt-* deps.
│   ├── mt-subtitles/   ASS/SRT parsing, encoding, dialogue model, processor.
│   ├── mt-discovery/   Filename → media identity (calls Python parser via mt-ml), hashing, TMDB.
│   ├── mt-fetch/       Providers, download, validate, score, align (ilass + xcorr), style classifier.
│   ├── mt-media/       FFmpeg extract/mux, font checks, file operations.
│   ├── mt-ml/          PyO3 embedded-CPython bridge → the Python package. THE boundary.
│   ├── mt-pipeline/    Orchestration: stages, GPU worker (serialised), progress events, proper nouns.
│   └── mt-cli/         clap CLI + ratatui TUI consuming the ProgressEvent stream.
├── movie_translator/               Python ML backend (importable package)
│   ├── translation/    Allegro BiDi model + Apple Translation (swift bridge) + sentence merger.
│   ├── ocr/            Vision OCR, PGS extractor, burned-in extractor, frame extractor.
│   ├── inpainting/     LaMa-based burned-in subtitle removal.
│   ├── identifier/     guessit/aniparse filename parser.
│   ├── ffmpeg.py, logging.py, types.py
│   └── tests/          + per-subpackage tests/ (pytest)
├── vendor/ilass/                   Git submodule — DP subtitle alignment engine (built by `just build`).
├── benchmarks/                     Translation-quality benchmarks (sacrebleu).
├── custom_model/                   Research/planning only — NOT wired into the pipeline.
├── docs/
│   ├── research/                   Custom-translation-model design (research only).
│   └── superpowers/{plans,specs}/  Historical design docs; rust-rewrite spec is the architecture record.
├── models/                         git-lfs translation model files.
├── justfile                        Tooling entry point (`just` to list).
├── Cargo.toml / Cargo.lock         Rust workspace.
├── pyproject.toml / uv.lock        Python deps (uv + ruff + ty + pytest).
├── rust-toolchain.toml             Pinned Rust channel (single source of truth, local + CI).
├── .python-version                 Python pin.
├── Brewfile                        macOS system tools.
├── conftest.py                     Shared pytest fixtures.
├── README.md                       User-facing docs.
├── AGENTS.md                       This file.
└── .claude/agents/                 Subagent definitions (tracked).
```

## Gotchas list

Every entry below cost real debugging time. Read once, save hours later.

### PyO3 / embedded-CPython boundary (`crates/mt-ml`)

- **`PYO3_PYTHON` must point at `.venv/bin/python`** at build time. The
  justfile exports it; CI sets it via env. Build against the system
  python and the embedded interpreter won't see torch/transformers and
  `import movie_translator`'s deps fail at runtime.
- **`multiprocessing.set_executable` is pinned to the venv python** at
  interpreter init (`init_python_runtime` in `backend.rs`). Without it,
  Python's spawn-start workers re-exec the `movie-translator` binary with
  a `-c <boilerplate>` argv, which clap then rejects — manifests as
  bizarre CLI errors from inside an ML stage.
- **Linux needs `LD_LIBRARY_PATH` to include libpython's dir** at
  runtime (PyO3 dynamically links libpython; macOS uses rpath and works
  without it). CI resolves it via `sysconfig LIBDIR`. A binary that
  builds fine but dies with "cannot open shared object file
  libpython…" at startup is this.
- **Don't hold the modules mutex across a re-entrant backend call.**
  `with_modules` clones the module handles under the lock then releases
  before running `f`, because importing/calling Python can re-enter the
  backend (translate → model_cache → modules). Holding the lock
  deadlocks. (Fixed once already — `e427aaa`.)
- **Python stderr is redirected** to `.translate_temp/python.stderr.log`
  (or `$MT_PYTHON_STDERR_LOG`). If an ML stage "fails silently," that
  log has the traceback, not the Rust logs.

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

### ML stages (`movie_translator/` Python package)

- **`transformers`/`tqdm` noise is silenced at init** — if you need the
  warnings for debugging, that suppression is in `init_python_runtime`.
- **OCR is Apple-Silicon-only** (Apple Vision). Burned-in OCR and
  `--ocr-language` don't work on Linux; the real-model tests are
  `#[ignore]`'d so CI (Linux + macOS, no GPU/model) stays green.
- **The Apple Translation backend builds a Swift bridge** from source on
  first use (`movie_translator/translation/swift/translate_bridge`,
  gitignored). macOS 26+ only.
- **`custom_model/` and `docs/research/` are research only.** Not wired
  into the pipeline. Don't treat them as live code paths — see
  `[[project_custom_model]]`.

### Build / toolchain / CI

- **`rust-toolchain.toml` is the single source of truth** for the
  compiler version (local + CI both read it via `rustup show`). Bump
  deliberately.
- **CI skips the git-lfs model fetch on purpose** (`lfs: false`) — the
  real-model tests are `#[ignore]`'d and skipping LFS avoids the repo's
  bandwidth quota. Don't "fix" CI by re-enabling LFS.
- **`uv sync` must run before any cargo command** in CI and locally —
  PyO3 reads `PYO3_PYTHON` (the venv python) at link time, so the venv
  must exist first.
- **`.translate_temp/` and `test_workdir/` are scratch** (gitignored).
  `--keep-artifacts` populates the former for debugging.

### Agentic workflow rules

- **Parallelize by DISJOINT file lanes.** Three non-overlapping lanes:
  Rust crates (`crates/**`), Python ML backend (`movie_translator/**`),
  tooling/docs (`justfile`, `.github/**`, `docs/**`, `scripts/**`,
  `pyproject.toml`, `conftest.py`). Fan out one agent per lane; they
  never touch each other's files. Same-file work is serial.
- **Serialize GPU + outward actions.** GPU is a single shared resource
  (see above). `--in-place`, `git lfs` ops, and looped provider fetches
  are destructive/outward — confirm, don't parallelize.
- **Named reusable workflows live in `.claude/workflows/`** — currently
  `review-changes` (read-only 3-lane diff review) and `benchmark-audit`
  (read-only benchmark-history audit). Invoke via the Workflow tool.
- **Verify via the gate chain, cite evidence.** Run the actual command
  (`just check` / `just test` / `just py-test`) and quote its output;
  never assert "done" or "should work" without having run it.
- **Hooks enforce, prose advises.** `.claude/hooks/` make two rules
  deterministic: `post-tool-use.sh` auto-formats every edited file
  (rustfmt / ruff) and surfaces syntax/lint errors; `stop-gate.sh`
  blocks turn-end on a failing fast gate (`cargo fmt --check` + `ruff
  check`) whenever the tree is dirty. The slow gates (clippy, cargo
  test, pytest, ty) stay at `just check`/`test`/`py-test` + CI. Hooks
  are snapshotted at session start — after editing them, reload
  (`/hooks` or restart) before relying on them.
- **No code index — plain grep/ripgrep is the default.** For structural
  Rust edits across crates, prefer compiler-driven refactors (rename via
  the type system, then `just check`) over text munging.

## Subagent index

Tracked under `.claude/agents/`. Each is a Claude Code subagent
definition — invoke via the Agent tool. Reach for these when the task
has the "spans several crates/tools, has multiple gotchas" character;
for one-shot file edits or simple queries, use the main agent directly.

| Subagent                       | When to use                                                                                                                                                  |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **`gate-verify`**              | After edits, want to ship-check: runs `just check` + `test` + `py-test`, parses each, reports green or the exact failing gate/test with output. Read-only.    |
| **`pyo3-bridge-doctor`**       | The Rust↔Python embedded-CPython boundary is broken: import failures, `PYO3_PYTHON` mismatch, libpython/`LD_LIBRARY_PATH`, multiprocessing-spawn crashes.    |
| **`ml-stage-debug`**           | A translation / OCR / inpainting stage misbehaves at runtime. Reproduces the Python side in isolation to split ML bug from Rust-bridge bug.                   |
| **`subtitle-fetch-align-debug`** | Fetched subs are wrong/rejected/mis-timed: provider issues, validation scoring, ilass/xcorr alignment, dialogue classification, offset correction.          |
| **`benchmark-runner`**         | Run the translation-quality benchmarks and/or audit stored results in git for regressions after a refactor.                                                  |
