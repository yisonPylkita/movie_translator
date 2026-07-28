# AGENTS.md — movie_translator

One-shot orientation. English→Polish video subtitle translator (MKV/MP4).
Produces files with as many Polish subtitle tracks as possible: identify media,
extract English (text/OCR/PGS), fetch existing Polish subs, validate & align,
fall back to AI translation, mux back.

## Architecture

**Pure Rust.** Cargo workspace (`crates/mt-*`) → `movie-translator` binary.
Zero Python/PyO3/venv. Apple Translation (macOS 26+, Swift bridge) +
Apple Vision OCR (macOS) + pure Rust Telea inpainting.
Filename parsing: `anitomy-pure` + regex. All ML inference native.

## Essential commands

| Command | Purpose |
| --------- | --------- |
| `just check && just test` | Full gate: clippy --workspace --all-targets -D warnings + nightly fmt --check + import check + all tests. Cite output. |
| `just run <file/dir> [flags]` | Translate (default). Flags: --dry-run, --no-fetch, --model apple, --workers N, --batch-size N, --inpaint, --in-place, --keep-artifacts, --hardsub-ocr, --force, --transcribe [--transcribe-engine {apple,whisper}] |
| `just extract <file/dir> [flags]` | Pull subs only, no translation. |
| `just anime-dl "<name>" [--out DIR] [--json PATH]` | Download anime season. --file watchlist.txt for batch. One title per line, optional episode filter, # comments. |
| `just setup` | Submodules + release build. Idempotent. `just brew` first on macOS. |
| `just check-imports` | ast-grep import hygiene scan (rule at `.pi/rules/ast-grep-rules/rules/import-function-over-path.yml`). |
| `just fix` | Auto-format + clippy fix + Cargo.toml sort. |

## File map

```
crates/
  mt-core/      Error, types, context, exec, identity
  mt-subtitles/ ASS/SRT parse/encode, dialogue model, processor
  mt-discovery/ Filename→media ID (anitomy + regex), hashing, TMDB
  mt-fetch/     Providers, download, validate, score, align (ilass + xcorr), style classifier
  mt-media/     FFmpeg mux/extract, font checks, PGS parser
  mt-ml/        ML: Apple Translation, Apple Vision OCR, Telea inpainting
  mt-pipeline/  Orchestration, GPU worker (serialised), progress, proper nouns
  mt-cli/       clap CLI, ratatui TUI. Two bins: movie-translator + anime-dl
vendor/ilass/   DP subtitle alignment (submodule, built by setup)
benchmarks/     ASR + translation benchmarks
docs/           research/ + superpowers/{plans,specs} (rust-rewrite spec = architecture record)
scripts/        Utility scripts
```

## Critical invariants

### GPU — one worker, serialised

Single tokio task (`worker.rs`). Every translate/OCR/inpaint `await`s completion
before next pull. Files parallelise (discovery/fetch/validation/mux) but GPU
stages funnel through one worker. `GpuExecutor` methods only safe from
`spawn_blocking` threads; never call off a runtime worker thread.

### Subtitles — fetch, validate, align

- **Dialogue detection** structural (position ratio, length, density). NOT keyword-based. Don't add keyword lists.
- **Validation** timing-overlap score ≥ 0.8 kept as separate track (user wants all viable tracks). Leading signs/karaoke before first dialogue line can skew score — known weak spot.
- **Alignment** ilass DP (primary), cross-correlation (fallback). Static offsets common (1-3 s or 60-90 s). Observed constant Polish-sub offset on Konosuba S1E1.
- **Providers** rate-limited: AnimeSub, Podnapisi, NapiProjekt, OpenSubtitles. `rate_limiter.rs` + `retry.rs` — backoff required.

### ML platform constraints

- **OCR:** macOS-only (Apple Vision via Swift bridge). Linux errors gracefully.
- **Apple Translation:** macOS 26+. Swift bridge compiled on demand (gitignored).
- **Inpainting:** pure Rust Telea, any platform.

### Build / toolchain

- `rust-toolchain.toml` = single SOT for compiler version (local + CI). Bump deliberately.
- `.translate_temp/` and `test_workdir/` = scratch (gitignored). `--keep-artifacts` populates former for debugging.

## Import hygiene (MUST)

Every qualified path call (`std::process::exit`, `tracing::info!`, `serde_json::from_str`)
must have a `use` import and short-form call.

```rust
// Correct
use tracing::info;
info!("done");

// Wrong
tracing::info!("done");
```

**Exceptions** (keep fully-qualified): derive macros (`#[from]`, `#[source]`, `#[serde]`, `#[clap]`),
type aliases (`pub type Result<T> = std::result::Result<T, MtError>`),
doc-comment backtick refs.

**After any import edit, run `cargo +nightly fmt`** (sorts imports automatically).

Checked by `just check` via ast-grep rule. Note: rule flags some idiomatic one-offs
(`Vec::new()`, `Path::new()`) — will be refined later.

### No unnecessary type annotations

```rust
// Bad
let mut rebuilt: Vec<String> = Vec::with_capacity(argv.len());
// Good
let mut rebuilt = Vec::with_capacity(argv.len());
```

Exceptions: `collect()` needs hint (`let x: HashSet<_> = expr.collect()`),
serde needs target type (`let x: SomeType = serde_json::from_str(json)?` or turbofish).

## Agentic workflow & delegation contract

- **Parent (top-level assistant) = orchestrator only.** Delegates all mutations to subagents. May use orchestration/status/wait/supervisor controls + final synthesis. Must NOT use edit/write/bash to mutate.
- **Child (subagent) = executor only.** Uses tools directly in assigned exclusive scope. Must NOT recursively delegate. See global `~/.pi/agent/AGENTS.md` for full policy.
- **Parallelise by disjoint file lanes:** Rust crates (`crates/**`) + tooling/docs (`justfile`, `.github/**`, `docs/**`). Never same-file.
- **Serialize GPU + outward actions.** `--in-place`, `git lfs`, looped provider fetches — confirm, don't parallelise.
- **Verify via gate chain, cite evidence.** Never assert "done" without `just check` / `just test` output.
- **No code index** — grep/ripgrep. Prefer compiler-driven refactors.
- **One writer per worktree.** Max 3 concurrent children. Shared manifests, lockfiles, CI/toolchain, README.md, AGENTS.md, design docs, foundational cross-crate APIs require explicit ownership. Stop on dirty overlap or unnamed shared-file changes.
- **Handoff:** one outcome, exact paths, done criteria, constraints, focused checks, commit/push permission (default: neither). Report changed paths, results, risks, assumptions, unmade decisions.

## Project agents

| Agent | When to use |
| ------- | ------------- |
| `fast_explorer` | Read-only repo evidence & impact mapping |
| `spec_analyst` | Read-only design, pipeline, platform, benchmark, acceptance analysis |
| `bounded_worker` | Exclusive-scope implementation + focused tests |
| `stage_owner` | Broader package, sole writer, independent review, one verify pass |

## Specialist skills

| Skill | When to use |
| ------- | ------------- |
| `gate-verify` | Run `just check + test`, report exact failing gate/test. Read-only. |
| `ml-stage-debug` | Diagnose translation/OCR/inpaint Rust/Swift bridge failures |
| `subtitle-fetch-align-debug` | Diagnose provider, validation, ilass/xcorr alignment, dialogue classification |
| `benchmark-runner` | Audit stored ASR/translation benchmark regressions |
