# Decisions

> Owner: parent orchestrator. Append after accepted decisions.
> Format: date, decision, reason, rejected alternatives, consequences.

## 2026-06 — Rust rewrite from Python/PyO3

**Decision:** Rewrite entire pipeline in pure Rust.

**Reason:** Eliminate Python/PyO3 dependency, improve performance, simplify deployment.

**Alternatives rejected:**

- Keep PyO3 bridge: maintenance burden, two-language complexity.
- Go: ecosystem mismatch for ML/media tooling.

**Consequences:**

- All ML runs natively (Apple Translation Swift bridge, Apple Vision OCR, Telea inpainting).
- No embedded CPython. No Python packages.
- `pyo3-bridge-doctor` skill deprecated, later removed.

## 2026 — ilass as primary alignment

**Decision:** Use ilass DP subtitle alignment as primary method, cross-correlation as fallback.

**Reason:** ilass handles non-linear timing drift better than simple offset detection.

**Alternatives rejected:**

- Cross-correlation only: fails on non-linear drift.
- FFmpeg-based alignment: less precise for subtitle timing.

**Consequences:**

- `vendor/ilass/` submodule required, built by `just setup`.
- Static offsets (1-3s, 60-90s) still common, handled by fallback.

## 2026 — Structural dialogue detection

**Decision:** Classify subtitle dialogue by structural features (position ratio, length, density).

**Reason:** Keyword lists are fragile across languages and genres.

**Alternatives rejected:**

- Keyword-based: fails on translated content, songs, genre shifts.
- ML classifier: unnecessary complexity for a structural problem.

**Consequences:**

- Leading signs/karaoke before first dialogue line can skew scores — known weak spot.
- Do not add keyword lists for dialogue detection.

## 2026 — GPU serialization

**Decision:** Single tokio task for all GPU work (translate, OCR, inpaint).

**Reason:** Avoid race conditions on Apple Neural Engine / GPU resources.

**Alternatives rejected:**

- Parallel GPU workers: resource contention, non-deterministic failures.
- Per-file GPU parallelism: files already parallelize at discovery/fetch/mux level.

**Consequences:**

- `GpuExecutor` methods only safe from `spawn_blocking` threads.
- Files parallelize but GPU stages serialize through one worker.
