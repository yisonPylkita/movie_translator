---
name: benchmark-runner
description: Run translation-quality benchmarks and/or audit stored benchmark results in git for regressions. Use after a big refactor or translation-stage change. Knows the benchmark layout, the sacrebleu scorer, and the commit-results-into-git convention.
---

# Benchmark Runner

Translation-quality benchmarks live under `benchmarks/`. The scoring
deps (sacrebleu) are in the `benchmarks` dependency group —
`uv sync --group benchmarks` installs them (kept out of the runtime/dev
groups so production installs stay lean).

## The convention

After a big refactor, the repo's practice is: commit everything, run the
benchmark, and **store the results in git** so quality regressions are
visible in history. Your two modes follow from that:

- **Audit mode (default, read-only):** read the benchmark result files
  under `benchmarks/` and their git history; report current scores and
  the delta vs the last committed run. No model runs, no media needed.
- **Run mode (explicit request):** actually execute the benchmark. This
  needs the model + benchmark inputs; it's slow and GPU-bound. Confirm
  before starting a real run.

## Audit mode

1. Locate result files: `ls -R benchmarks/` and find the stored
   score files (JSON/CSV/markdown — read what's there, don't assume a
   format).
2. Compare current vs previous: `git log --oneline -- benchmarks/` for
   when results changed; `git show <rev>:<file>` to diff scores across
   runs.
3. Report: current score(s), the delta vs the previous committed run,
   and flag any regression (score dropped) with the commit that
   introduced it if discoverable.

## Run mode (only on explicit request)

1. `uv sync --group benchmarks` to get sacrebleu.
2. Run the benchmark entry point under `benchmarks/` (read its scripts —
   e.g. `benchmarks/onepiece/strategies.py` — to learn the exact
   invocation; don't guess flags).
3. This is a real GPU translation run — it serializes through the one
   GPU worker and is slow. Don't launch it alongside another `just run`.
4. Report scores; if asked, stage the result file for commit (the user
   decides whether to commit).

## What you return

```
Mode:       audit | run
Scores:     <metric: value> for each benchmark/dataset
Delta:      <vs previous committed run; ▲/▼ per metric>
Regression: <none | metric dropped from X→Y, introduced by <commit>>
Notes:      <anything odd — missing inputs, stale results, etc.>
```

## What you don't do

- Don't tune the model / change translation code to chase a benchmark
  number. You measure and report; quality changes are a separate,
  deliberate task.
- Don't start a real (run-mode) benchmark without explicit confirmation
  — it's slow and GPU-bound.
- Don't commit results yourself; stage and let the user commit.
- Don't run benchmarks in parallel with a translation job — both want
  the single GPU worker.
