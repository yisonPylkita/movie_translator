# Current Plan

> Owner: parent orchestrator. Update when objective changes or items complete.

## Objective

Context/token optimization — reduce per-request overhead and tool-output bloat.

## Completed

- Context optimization package: standardized agent handoffs, project memory files, tool-output storage convention, global prune config enabled.
- Removed deprecated `pyo3-bridge-doctor` skill.
- Anime-dl robustness overhaul landed on main (2026-07-31): host abstraction (`hosts.rs`), manifest v2 + atomic save (`manifest.rs`), download validation (`validator.rs`), downloader rewrite with retry/backoff, circuit breaker, validation gate, quarantine, cancellation + RAII cleanup, TUI restructure (`tui_download`/`scoreboard`/`stream`/`timeline` removed → `ui_model`/`ui_render`), plain output polish, userscript v4 (XSS-safe `textContent`, `@grant none`), design spec `docs/superpowers/specs/2026-07-31-anime-dl-robustness-design.md`. `just check` PASS, `just test` PASS (799 passed, 0 failed, 6 ignored env-dependent); independent review accepted, no blocker/HIGH.

## In Progress

- None.

## Next

- Measure token composition after optimization.
- Evaluate Hypa in replace mode if shell/test output still bloated.
- Consider ReadSeek if whole-file reads remain major cost.
- Benchmark 3 representative workloads (small fix, medium feature, repo-wide investigation).
- Investigate dood.yt host support (fetch/resolve currently unsupported by host abstraction).
- Wire per-host timeout profiles (engine currently uses fixed 30s startup timeout).
- Fix validate-only exit-code double-count on mixed runs; raise min-size flag floor above 1 MiB.
- Cargo-sort toolchain lane (`cargo sort --check` debt in Cargo.toml files).

## Blockers

- None.
