---
name: gate-verify
description: Run the full gate chain (just check, just test) to verify Rust changes are green. Use after editing code in this repo before claiming done. Parses each gate's output and identifies exactly what failed.
---

# Gate Verify

Run the full gate chain on this repo and report whether it's GREEN or RED.

## The gate chain

```text
just check → clippy -D warnings + cargo +nightly fmt --check
             + TOML/shell/Swift/JSON/import checks
just test  → cargo test --workspace
just fix   → format + clippy-fix + cargo sort
```

## Flow

1. **`just check`.** Bisect clippy vs fmt failures.
2. **`just test`.** Report failing test names.
3. **`cargo sort -w --check`.** Validate Cargo.toml dependency ordering.

## What you return

```text
just check:    <pass | fail>
just test:     <N passed | M failed>
deps sorted:   <pass | fail>
Overall:       GREEN | RED
```

Always cite the actual command output. Don't claim GREEN without
running all gates.
