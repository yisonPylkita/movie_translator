# Movie Translator Python → Rust Rewrite — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the movie_translator orchestration + logic layers from Python to a Rust Cargo workspace, keeping ML inference in Python as single-purpose CLI scripts, with the Python codebase as the correctness oracle until a final parity gate.

**Architecture:** Cargo workspace of focused crates (`mt-core` … `mt-cli`). Rust owns all orchestration, I/O, parsing, alignment, validation, and GPU-worker serialization (tokio). ML tasks (translate, OCR, inpaint, font-subset) are delegated to standalone Python CLI scripts spawned per stage-per-file. Built bottom-up; each module is ported test-first with its Python tests carried over, then verified against Python output.

**Tech Stack:** Rust 2021, tokio, clap, reqwest, serde/serde_json, ndarray + rustfft (xcorr), ratatui/indicatif (progress), thiserror/anyhow. Python 3.10+ (existing) for ML scripts.

**Reference convention:** Each port task cites the Python source file it reproduces. That source is the line-level spec — preserve its observable behavior. Port the matching Python tests (`*/tests/test_*.py`) into Rust tests as the acceptance bar. Do not invent new behavior; this is a behavior-preserving port.

**Spec:** `docs/superpowers/specs/2026-05-27-rust-rewrite-design.md`

**Branch:** `rust-rewrite` (work unattended to completion; commit after every task).

---

## Layout

```
Cargo.toml                 # workspace
crates/
  mt-core/                 # domain types, errors
  mt-subtitles/            # ASS/SRT parse + write + processing
  mt-discovery/            # filename parse + media identify
  mt-fetch/                # providers, validate, score, align
  mt-media/                # ffmpeg/ffprobe, extract, mux, font detection
  mt-ml/                   # drives Python ML CLI scripts
  mt-pipeline/             # orchestration, stages, gpu worker, tokio
  mt-cli/                  # clap CLI, commands, progress
ml/                        # standalone Python ML CLI scripts
  translate.py  ocr_pgs.py  ocr_burned_in.py  inpaint.py  font_subset.py
```

Python package stays in place under `movie_translator/` until cutover.

---

## Phase 0 — Scaffold, ML scripts, parser spike

### Task 0.1: Workspace scaffold
**Files:** Create `Cargo.toml` (workspace), `crates/*/Cargo.toml`, `crates/*/src/lib.rs` (empty) for all 8 crates.
- [ ] Create workspace `Cargo.toml` with `[workspace] members = ["crates/*"]`, shared `[workspace.dependencies]` (tokio, serde, serde_json, thiserror, anyhow, reqwest, clap, tracing).
- [ ] Create each crate with `cargo new --lib`, wire internal deps (mt-cli depends on mt-pipeline → … → mt-core).
- [ ] Run `cargo build` — expect success (empty crates).
- [ ] Commit: `chore: rust workspace scaffold`.

### Task 0.2: ML CLI scripts (thin wrappers over existing Python modules)
**Files:** Create `ml/translate.py`, `ml/ocr_pgs.py`, `ml/ocr_burned_in.py`, `ml/inpaint.py`, `ml/font_subset.py`.
Each: parse argv + read JSON request on stdin, call the existing `movie_translator.*` module, write JSON result to stdout. One task per script. Define the JSON contract per script (documented in a module docstring) — this contract is consumed by `mt-ml`.
- [ ] `translate.py`: stdin `{lines:[{start_ms,end_ms,text}], model, device, batch_size}` → stdout `{translations:[str]}`. Wraps `translation.translator.SubtitleTranslator`.
- [ ] `ocr_pgs.py`: argv `--video PATH --track N` → stdout `{srt_path, results:[{timestamp_ms,text,boxes}]}`. Wraps `ocr.pgs_extractor`.
- [ ] `ocr_burned_in.py`: argv `--video PATH` → stdout `BurnedInResult` JSON. Wraps `ocr.burned_in_extractor`.
- [ ] `inpaint.py`: argv `--video PATH --mask ... --out PATH`. Wraps `inpainting.video_processor`.
- [ ] `font_subset.py`: argv `--fonts ... --text ... --out ...`. Wraps `fonts.py` subsetting path.
- [ ] Smoke-test each script standalone with a tiny input; commit per script: `feat(ml): <task> cli script`.

### Task 0.3: ASS parser spike (decide hand-roll vs `subparse`)
**Files:** `crates/mt-subtitles/examples/spike_handroll.rs`, `examples/spike_subparse.rs`; corpus from existing `.ass` test files.
- [ ] Collect test `.ass` files: `find movie_translator -name '*.ass'` and any in `benchmarks/`.
- [ ] Spike A: parse+rewrite with a minimal hand-rolled ASS parser; measure round-trip fidelity (events, styles, timing, override tags preserved).
- [ ] Spike B: same with `subparse` crate.
- [ ] Compare against pysubs2 round-trip (run pysubs2 on same files as ground truth).
- [ ] Record decision in a comment block in `mt-subtitles/src/lib.rs` and proceed with the winner in Phase 2.
- [ ] Commit: `chore(subtitles): ASS parser spike + decision`.

---

## Phase 1 — mt-core (domain types)

### Task 1.1: Port domain types
**Files:** Create `crates/mt-core/src/{types.rs,context.rs,error.rs,lib.rs}`.
**Reference:** `movie_translator/types.py`, `movie_translator/context.py`.
- [ ] Port: `DialogueLine{start_ms:i64,end_ms:i64,text:String}`, `SubtitleFile`, `BoundingBox{f32}`, `OCRResult`, `BurnedInResult`, `OriginalTrack`, `FontInfo`, `FetchedSubtitle`, `PipelineConfig` (with serde derives). Constants: `NON_DIALOGUE_STYLES`, `POLISH_CHARS`, `POLISH_CHAR_MAP`.
- [ ] Port `replace_polish_chars`; test: maps `ąćęł…` → `acel…`, leaves ASCII unchanged.
- [ ] Define `MtError` enum (thiserror) covering IO, parse, subprocess, network categories.
- [ ] `cargo test -p mt-core` passes. Commit: `feat(core): domain types + errors`.

---

## Phase 2 — mt-subtitles

Reference: `movie_translator/subtitles/{extractor.py,processor.py,_pysubs2.py}` and `subtitle_fetch/encoding.py`.

### Task 2.1: ASS/SRT read+write (winner of 0.3)
- [ ] Implement parse → in-memory model (script info, styles, events) and serialize back. Round-trip test against every corpus `.ass`/`.srt`: re-serialized output semantically equals pysubs2 output (events count, timings ±0, text, style names).
- [ ] Commit: `feat(subtitles): ass/srt read+write`.

### Task 2.2: Encoding detection
**Reference:** `subtitle_fetch/encoding.py`.
- [ ] Port charset detection (chardet-equiv: use `chardetng` or `encoding_rs`); test against the same fixtures the Python tests use. Commit.

### Task 2.3: Subtitle processing
**Reference:** `subtitles/processor.py` (dialogue extraction, non-dialogue filtering via `NON_DIALOGUE_STYLES`, line merging).
- [ ] Port each public function with its Python test cases. Commit per function group.

---

## Phase 3 — mt-discovery

Reference: `movie_translator/discovery.py`, `identifier/{hasher.py,napihash.py,parser.py,identify.py,metadata.py,tmdb.py}`.

### Task 3.1: Hashes (pure, exact)
- [ ] `compute_oshash` (OpenSubtitles 64-bit sum hash) — port `identifier/hasher.py` exactly; test with a known fixture file + expected hex (generate expected from Python).
- [ ] `napihash` — port `identifier/napihash.py`; test against Python output.
- [ ] Commit: `feat(discovery): oshash + napihash`.

### Task 3.2: Filename parsing — Python CLI tool (REVISED 2026-05-27)
`identifier/parser.py` wraps `guessit` + `aniparse`, large Python rule engines with no faithful Rust equivalent. Per the user's CLI-tools-over-Rust-port principle (same as ML), keep parsing in Python behind a thin single-purpose CLI script; Rust orchestrates.
- [ ] Create `ml/parse_filename.py`: argv/stdin `{filename, folder_name?}` → stdout JSON `{title, parsed_title, year, season, episode, media_type, is_anime, release_group}` (the fields `parse_filename` produces). Wraps `identifier.parser.parse_filename`.
- [ ] In `mt-discovery`, a `parse_filename(filename, folder) -> ParsedName` that spawns the script (reuse the mt-ml runner once it exists, or a local subprocess helper) and deserializes. Test with a `--echo`/fixture mode or `#[ignore]` integration test plus a unit test of the deserialization.
- [ ] Commit: `feat(discovery): filename parse via python cli tool`.

### Task 3.3: TMDB lookup + identify orchestration
**Reference:** `identifier/tmdb.py`, `identify.py`, `metadata.py`.
- [ ] Port HTTP TMDB client (reqwest) — mock HTTP in tests. Port `identify_media` flow. Commit.

---

## Phase 4 — mt-media

Reference: `movie_translator/{ffmpeg.py,extract.py,fonts.py}`, `video/operations.py`, `subtitles/extractor.py`, `stages/mux.py`.

### Task 4.1: ffmpeg/ffprobe wrappers
- [ ] Port `get_ffmpeg/get_ffprobe` (locate static-ffmpeg equivalent — use a vendored/`which` lookup), and the probe/exec helpers. Test by probing a tiny fixture mkv (add one under `crates/mt-media/tests/fixtures/`). Commit.

### Task 4.2: Track extraction
**Reference:** `extract.py`, `subtitles/extractor.py`, `stages/extract_*.py`.
- [ ] Port subtitle/track extraction (text tracks via ffmpeg; PGS/burned-in route to `mt-ml` scripts later). Test against fixture. Commit.

### Task 4.3: Font detection (subsetting stays in ml/font_subset.py)
**Reference:** `fonts.py`.
- [ ] Port embedded-font listing + Polish-support detection + fallback-font selection. The actual subset/embed call delegates to `ml/font_subset.py` via `mt-ml`. Test detection logic against fixtures. Commit.

### Task 4.4: Mux
**Reference:** `stages/mux.py`, `stages/create_tracks.py`.
- [ ] Port track-assembly + mux command construction; test command строки against expected argv (port `tests/test_mux.py`, `test_create_tracks.py`). Commit.

---

## Phase 5 — mt-fetch

Reference: `movie_translator/subtitle_fetch/*`.

### Task 5.1: Providers (reqwest)
**Reference:** `subtitle_fetch/fetcher.py`, `rate_limiter.py`, `retry.py`, providers for AnimeSub/Podnapisi/NapiProjekt/OpenSubtitles.
- [ ] Port each provider client; mock HTTP in tests. Port rate limiter + retry/backoff with deterministic time injection. Commit per provider.

### Task 5.2: Validation + scoring
**Reference:** `validator.py`, `scoring.py`, `style_classifier.py`. **Note:** see memory `project_validation_bug` — validation must handle files with non-dialogue events before the first dialogue. Add a test for that case.
- [ ] Port timing-based validation + scoring; carry Python test cases + add the non-dialogue-prefix regression test. Commit.

### Task 5.3: Alignment
**Reference:** `align.py` (cross-correlation), `align_ilass.py` (ilass subprocess), `types.py`.
- [ ] Port xcorr alignment using `ndarray` + `rustfft`; test against Python output on a fixture pair (including the static-offset case from memory `project_subtitle_alignment`).
- [ ] Port ilass-subprocess driver + the cross-correlation fallback selection logic. Commit.

---

## Phase 6 — mt-ml (script drivers)

### Task 6.1: Script runner + JSON contracts
**Files:** `crates/mt-ml/src/{lib.rs,translate.rs,ocr.rs,inpaint.rs,fonts.rs}`.
- [ ] Generic `run_script(name, request: impl Serialize) -> Result<Response>` that spawns `python ml/<name>.py`, writes JSON stdin, reads JSON stdout, surfaces nonzero exit + stderr as `MtError`.
- [ ] Typed wrappers matching each script's contract from Task 0.2 (`translate`, `ocr_pgs`, `ocr_burned_in`, `inpaint`, `font_subset`).
- [ ] Integration test: round-trip a tiny `translate` request through the real script (mark `#[ignore]` if it needs models; provide a `--dry-run`/echo mode in the script for CI). Commit.

---

## Phase 7 — mt-pipeline

Reference: `movie_translator/{pipeline.py,async_pipeline.py,gpu_queue.py,progress.py}`, `stages/*`.

### Task 7.1: GPU worker + serialization (tokio)
**Reference:** `gpu_queue.py`.
- [ ] Replace the Python async GPU worker with a tokio task + `Semaphore(1)` serializing GPU-bound work (translate/ocr/inpaint script spawns); I/O-bound work unbounded. Submit/await via channels. Tests: concurrent submissions serialize on the GPU permit; ordering/back-pressure as in Python. Commit.

### Task 7.2: Stages
**Reference:** each `stages/*.py` (identify, extract_english, extract_ref, fetch, translate, create_tracks, mux, extract_english).
- [ ] Port each stage as a function over `PipelineContext`, calling the relevant crate. Port each stage's tests. Commit per stage.

### Task 7.3: Pipeline orchestration
**Reference:** `pipeline.py`, `async_pipeline.py`, `context.py`.
- [ ] Port the progressive-context pipeline + async driver (tokio), wiring stages in order with the worker pool (`workers` config). Integration test on a fixture mkv end-to-end with ML scripts in dry/echo mode. Commit.

---

## Phase 8 — mt-cli

Reference: `movie_translator/{main.py}`, `commands/*`.

### Task 8.1: CLI args (clap)
- [ ] Port arg parsing for `translate`, `extract` subcommands matching `commands/*` + `main.py`. Test parsing → config. Commit.

### Task 8.2: Commands
**Reference:** `commands/{translate_cmd.py,extract_cmd.py,common.py}`.
- [ ] Port each command to invoke mt-pipeline. Commit per command.

### Task 8.3: Progress display (TUI)
**Reference:** `progress.py`, `tui_renderer.py`. **Note:** memory `project_logging_overhaul` — user wants a rich interactive TUI, not scrolling logs. Use `ratatui` (or `indicatif` if a full TUI is overkill). Match current display behavior.
- [ ] Port progress callbacks + renderer. Commit.

---

## Phase 9 — Parity gate + cutover

### Task 9.1: End-to-end parity vs Python
- [ ] Run `benchmarks/onepiece` BLEU (sacrebleu) using the Rust binary's translate path vs the Python path on the same corpus; assert BLEU delta within tolerance **±0.5 BLEU** (set here, per spec open item).
- [ ] Re-run the stored benchmark JSONs; Rust output must match Python within tolerance.
- [ ] Run any existing integration corpus end-to-end; diff output subtitle tracks (timing exact, text equal modulo model nondeterminism).
- [ ] Commit: `test: rust/python parity gate green`.

### Task 9.2: Cutover
- [ ] Repoint the `movie-translator` entry point to the Rust binary (build instructions in README; keep `ml/` Python scripts as the only Python runtime dep).
- [ ] Update README + pyproject (Python now only the ML scripts). Commit: `feat: cut over entry point to rust binary`.

---

## Self-Review notes
- **Spec coverage:** every boundary item (translation, ocr, inpaint, fonts → ml scripts; subtitles/fetch/discovery/media/pipeline/cli → crates) has a phase. Verification + cutover covered (Phase 9). ✓
- **Open items resolved here:** parity tolerance = ±0.5 BLEU (9.1); ASS parser = spike-then-pick (0.3); build order = bottom-up Phases 1→8. ✓
- **Risk tasks flagged inline:** guessit port (3.2 parity corpus), ASS fidelity (0.3/2.1), fonts kept in Python (0.2/4.3), xcorr (5.3). ✓
