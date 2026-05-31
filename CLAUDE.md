# CLAUDE.md — session bootstrap

Agent guidance for this repo lives in `AGENTS.md` — the subagent table,
the "how to do common things" workflows, the file map, and the
hard-won gotchas list. Read it; it's the real map. This file just
auto-loads it and pins the non-negotiables.

@AGENTS.md

## Always

- **Gate before "done".** Run `just check && just test && just py-test`
  before committing. Never claim work is done without running the gate
  and citing its output — evidence before assertions. (`just check` =
  clippy `-D warnings` + `cargo fmt --check` + ruff.)
- **One GPU — ML work is serialized; keep it that way.** Translation,
  OCR, and inpainting all run through a single tokio GPU worker
  (`crates/mt-pipeline/src/worker.rs`). Files process in parallel, but
  GPU stages run one at a time. Don't add code paths that call
  `mt_ml::{translate,ocr_*,inpaint}` concurrently, and don't launch two
  real `just run` jobs against the same GPU at once.
- **Parallelize only DISJOINT-file lanes.** Three lanes that never share
  files: Rust crates (`crates/**`), Python ML backend
  (`movie_translator/**`), tooling/docs (`justfile`, `.github/**`,
  `docs/**`, `scripts/**`, `pyproject.toml`). Fan out one agent per lane;
  same-file work is serial.
- **ML stays Python; Rust owns orchestration.** The split is deliberate
  (type-safety in the pipeline, ML in single-purpose Python). Don't
  reimplement translation/OCR/inpainting in Rust. New ML behavior goes in
  the `movie_translator` package, exposed through `crates/mt-ml`.
- **Never break the PyO3 build contract.** `crates/mt-ml` embeds CPython;
  `PYO3_PYTHON` must point at `.venv/bin/python` (the justfile + CI set
  this). `just deps` creates the venv before any cargo build links
  against libpython.
- **Confirm before destructive / outward-facing actions:** `just run
  --in-place` (overwrites original videos), `git lfs` model
  pulls/pushes, and anything that hits the live subtitle providers in a
  loop (AnimeSub / Podnapisi / NapiProjekt / OpenSubtitles are
  rate-limited — don't hammer them). Ask first; use `--dry-run` to
  preview without touching originals.
