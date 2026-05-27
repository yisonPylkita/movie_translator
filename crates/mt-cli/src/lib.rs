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

/// Our own crates, raised to `debug` under `--verbose`.
const OUR_CRATES: &[&str] = &[
    "mt_core",
    "mt_subtitles",
    "mt_discovery",
    "mt_media",
    "mt_fetch",
    "mt_ml",
    "mt_pipeline",
    "mt_cli",
    "movie_translator",
];

/// Build the default tracing filter directive string.
///
/// Third-party crates (html5ever/hyper/rustls/reqwest/...) are capped at `warn`
/// so `--verbose` doesn't drown the terminal in their internal DEBUG spam. Our
/// own crates go to `debug` under `--verbose`, else `info`. `RUST_LOG` (read by
/// [`tracing_subscriber::EnvFilter::try_from_default_env`]) overrides this.
fn default_filter(verbose: bool) -> String {
    let our_level = if verbose { "debug" } else { "info" };
    let mut directive = String::from("warn");
    for crate_name in OUR_CRATES {
        directive.push_str(&format!(",{crate_name}={our_level}"));
    }
    directive
}

/// Initialise the tracing subscriber once. `verbose` raises our crates to DEBUG.
///
/// Mirrors `set_verbose(args.verbose)` in the Python handlers. Logs go to
/// stderr so stdout stays clean for the summary line. Uses an `EnvFilter` so
/// third-party crates stay capped at `warn` (no html5ever/hyper/rustls DEBUG
/// flood under `-v`); honours a `RUST_LOG` override.
pub fn init_tracing(verbose: bool) {
    TRACING_INIT.call_once(|| {
        let filter = tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(default_filter(verbose)));
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(std::io::stderr)
            .with_target(false)
            .try_init();
    });
}

#[cfg(test)]
mod tests {
    use super::default_filter;

    #[test]
    fn filter_caps_third_party_at_warn() {
        let verbose = default_filter(true);
        assert!(verbose.starts_with("warn"), "third-party default is warn");
        assert!(verbose.contains("mt_cli=debug"));
        assert!(verbose.contains("mt_pipeline=debug"));
        // No bare `debug` directive that would let html5ever/hyper through.
        assert!(!verbose.contains(",debug"));
    }

    #[test]
    fn filter_non_verbose_uses_info_for_our_crates() {
        let quiet = default_filter(false);
        assert!(quiet.starts_with("warn"));
        assert!(quiet.contains("mt_cli=info"));
        assert!(!quiet.contains("=debug"));
    }
}
