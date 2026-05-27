//! CLI entry point: argument parsing, command dispatch, progress UI.
//!
//! Port of `movie_translator/main.py` (subcommand routing) and the
//! `movie_translator/commands/` handlers. The binary is `movie-translator`
//! (see `src/main.rs`).

pub mod commands;
pub mod common;
pub mod progress;

use std::sync::Once;

static TRACING_INIT: Once = Once::new();

/// Initialise the tracing subscriber once. `verbose` raises the level to DEBUG.
///
/// Mirrors `set_verbose(args.verbose)` in the Python handlers. Logs go to
/// stderr so stdout stays clean for the summary line.
pub fn init_tracing(verbose: bool) {
    TRACING_INIT.call_once(|| {
        let level = if verbose {
            tracing::Level::DEBUG
        } else {
            tracing::Level::INFO
        };
        let _ = tracing_subscriber::fmt()
            .with_max_level(level)
            .with_writer(std::io::stderr)
            .with_target(false)
            .try_init();
    });
}
