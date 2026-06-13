# Progress — mt-pipeline import fixes

## Completed

**orchestrator.rs**:
- Added `use std::fs;`, `use tokio::{spawn, task};`, `use tokio::sync::mpsc;`, `use tracing::{debug, error, info, warn};`
- Changed all `tracing::info!` → `info!`, `tracing::warn!` → `warn!`, `tracing::error!` → `error!`, `tracing::debug!` → `debug!`
- Changed all `tokio::task::spawn_blocking(...)` → `task::spawn_blocking(...)`
- Changed all `tokio::spawn(...)` → `spawn(...)`
- Changed `tokio::sync::mpsc::unbounded_channel()` → `mpsc::unbounded_channel()`
- Changed all `std::fs::remove_dir_all/write/create_dir_all/read_dir/remove_dir` → `fs::*`
- Changed all `tempfile::tempdir()` → `tempdir()` (9 instances)
- Added `use tempfile::tempdir;` inside test module
- Added `use tokio::sync::mpsc;` inside test module

**worker.rs**:
- Added `use tokio::{spawn, select, task};`, `use std::sync::atomic::Ordering;`
- Changed `tokio::spawn` → `spawn`, `tokio::select!` → `select!`, `tokio::task::spawn_blocking` → `task::spawn_blocking`
- Changed `tokio::task::JoinHandle` → `task::JoinHandle`

**gpu.rs** — already clean (was using `use tracing::warn;`)

**proper_nouns.rs** — already clean (was using `use tracing::info;`)

**All stage files** (identify, extract_ref, extract_english, fetch, translate, create_tracks, mux, hardsub_ocr, transcribe):
- Changed all `tracing::info/warn/error/debug` → `info/warn/error/debug`
- Changed all `tempfile::tempdir()` → `tempdir()` with proper imports
- Changed `serde_json::from_str(...)` → `from_str(...)` with proper import
- Added `use std::fs;` where needed, changed all `std::fs::*` → `fs::*`
- Added `use std::fs;` inside test modules where `fs::` is used

## Verification
- `cargo clippy -p mt-pipeline`: No issues found
- `cargo test -p mt-pipeline`: 78 passed, 3 ignored
- `cargo fmt --check`: Clean
