---
name: pyo3-bridge-doctor
description: DEPRECATED — this project no longer uses Python/PyO3. All ML runs natively in Rust. The PyO3 dependency was removed. This skill exists for historical reference only.
---

# DEPRECATED

This project removed all Python code in a June 2026 refactor. The
entire `movie_translator/` Python package was deleted, the `backend.rs`
PyO3 bridge was removed, and `pyo3` was dropped from `Cargo.toml`.

There is no embedded CPython. There is no Python to debug.

Use `ml-stage-debug` for ML stage issues instead. All stages are pure Rust.
