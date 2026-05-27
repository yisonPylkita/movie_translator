# Movie Translator: Python → Rust Rewrite — Design

**Date:** 2026-05-27
**Status:** Approved design, pending implementation plan

## Motivation

The prototyping phase is complete. The goal of the rewrite is **type safety and
correctness**: the orchestration and logic layers (pipeline, subtitle parsing,
alignment, validation, filename discovery) are where bugs hide and where Rust's
compile-time guarantees pay off most. Numeric/model-bound ML work gains little
from a language change and has no mature Rust equivalent, so it stays in Python.

## Strategy

**Big-bang orchestration rewrite.** Rewrite the whole orchestration layer in Rust
at once rather than shipping a mixed binary. The existing Python codebase remains
runnable throughout as the **correctness oracle**; we cut over the entry point
only after Rust reaches parity. "Big-bang" refers to the cutover model, not the
testing discipline — each subsystem is built test-first and verified against the
Python behavior before moving on.

## The Rust / Python Boundary

### Stays Python (numeric / model-bound, no Rust equivalent)

- `translation/` — torch + transformers (NLLB / Allegro BiDi), Apple Translation
  backend, enhancements, sentence_merger, proper_nouns
- `ocr/` — pyobjc Vision (macOS-only), PGS extractor, burned-in extractor, opencv
  frame work
- `inpainting/` — LaMa (torch) + opencv
- font subsetting/embedding — fonttools (no clean Rust equivalent)

### Ports to Rust (logic-heavy, type-safety win)

- domain types and errors
- `subtitles/` — ASS/SRT parse + write + processing (pysubs2 replacement)
- `subtitle_fetch/` — providers (reqwest), validator, scoring, alignment
  (ilass subprocess + cross-correlation fallback), encoding detection
- `identifier/` + `discovery` — oshash, napihash, tmdb, guessit/aniparse filename
  parsing
- `ffmpeg`/`ffprobe` wrappers, extract, mux, font detection
- pipeline, stages, async orchestration (tokio)
- CLI, commands, TUI progress
- metrics

## ML Delegation: Single-Purpose CLI Scripts

**No persistent sidecar service.** Each ML task is a standalone, single-purpose
Python CLI script doing exactly one thing:

- `scripts/translate.py` — translate a batch of dialogue lines
- `scripts/ocr_pgs.py` — OCR a PGS subtitle track
- `scripts/ocr_burned_in.py` — OCR burned-in subtitles
- `scripts/inpaint.py` — inpaint a video region (subtitle removal)
- `scripts/font_subset.py` — subset/embed fonts (fonttools)

Each script reads input via CLI args + files + JSON on stdin and writes results
as JSON on stdout (or to an output file for binary/media payloads). These scripts
reuse the existing, working Python ML modules internally — they are thin CLI
wrappers, not rewrites.

**Invocation granularity: per stage-per-file.** One script invocation processes a
whole file's worth of work in a single batch (all dialogue lines, all frames), so
the torch model loads once per invocation rather than once per item. This matches
the current batch logic and keeps model-load overhead negligible.

## Orchestration & Concurrency — Fully in Rust

The GPU worker and serialization logic ports entirely to Rust:

- A tokio-based worker model replaces the Python `gpu_queue` async worker.
- GPU-bound work is serialized via a semaphore (the GPU is a single shared
  resource); concurrency for I/O-bound work (downloads, ffmpeg) is unconstrained.
- Rust workers spawn the Python ML scripts as subprocesses, feed them batched
  input, and parse their JSON output.
- Communication between the main process and workers uses native Rust mechanisms
  (channels, async tasks) — no Python-side coordination.

## Workspace Layout — Cargo Workspace

```
mt-core        domain types, models, errors
mt-subtitles   ASS/SRT parse + write + processing
mt-discovery   filename parse + media identify (oshash / napihash / tmdb)
mt-fetch       providers (reqwest), validate, score, align (ilass subproc + xcorr)
mt-media       ffmpeg / ffprobe wrappers, extract, mux, font detection
mt-ml          spawns + drives the Python ML CLI scripts
mt-pipeline    orchestration, stages, tokio async, GPU worker/serialization
mt-cli         clap CLI, commands, ratatui / indicatif progress
```

## Verification

The Python test suite and benchmarks are the behavioral specification.

- **Per subsystem:** port the relevant Python tests to Rust and make them pass
  before moving on.
- **Parity gate (final):** run `benchmarks/onepiece` (BLEU via sacrebleu) and the
  stored benchmark JSONs against Rust output. Rust must match Python within an
  agreed tolerance before cutover.
- The Python repo stays runnable until the parity gate passes; then the entry
  point swaps to the Rust binary.

## Top Fidelity Risks

1. **ASS parsing.** pysubs2 is mature (styles, override tags, embedded fonts).
   Hand-roll a parser or use `subparse`; verify round-trip against existing test
   `.ass` files. Round-trip fidelity is the acceptance bar.
2. **guessit / aniparse.** Large rule sets for filename parsing — highest
   porting-fidelity risk. Mitigate with a parity test-corpus harvested from
   Python output across many real filenames.
3. **Fonts.** Subsetting/embedding stays Python (`scripts/font_subset.py`);
   only detection ports to Rust.
4. **align.py cross-correlation.** numpy → `ndarray` / `rustfft`.

## Out of Scope

- Porting ML inference to Rust (candle/ONNX/objc2 Vision) — explicitly rejected.
- A persistent ML service — explicitly rejected in favor of CLI scripts.
- New features. This is a behavior-preserving port; the Python benchmarks define
  "correct."
