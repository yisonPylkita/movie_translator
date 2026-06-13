---
name: benchmark-runner
description: Audit stored benchmark results in git for regressions after a refactor. Pure Rust project — no Python needed.
---

# Benchmark Runner

ASR and translation benchmarks live under `benchmarks/`. This is a pure
Rust project — no Python/uv needed.

## Convention

Commit benchmarks results into git so quality regressions are visible in
history. Flag any score change > 2% as a potential regression.

## What you return

```
Benchmark:  asr | translation
Scores:     <table>
Delta:      <compare with stored results>
Overall:    no regression | regression in <metric>
```
