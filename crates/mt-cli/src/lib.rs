//! CLI entry point: argument parsing, command dispatch, progress UI.
//!
//! The binary is `movie-translator` (see `src/main.rs`).

pub mod commands;
pub mod common;
pub mod tui;
pub mod tui_download;

use std::io::stderr;
use std::sync::Once;

use mt_pipeline::ProgressSender;
use tracing_subscriber::EnvFilter;
use tui::TuiTracingLayer;

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
/// Logs go to stderr so stdout stays clean for the summary line. Uses an
/// `EnvFilter` so third-party crates stay capped at `warn` (no
/// html5ever/hyper/rustls DEBUG flood under `-v`); honours a `RUST_LOG`
/// override.
pub fn init_tracing(verbose: bool) {
    init_tracing_with(verbose, None);
}

/// Like [`init_tracing`] but also installs a [`tui::TuiTracingLayer`] that
/// forwards tracing events into the TUI's progress channel as `Log` events.
///
/// Without a TUI sender this is identical to [`init_tracing`]: stderr-only,
/// info/debug filtered as configured. With a TUI sender, tracing events ALSO
/// reach the TUI's log pane (the fmt-stderr layer is dropped when the TUI is
/// active to avoid double-painting the alternate screen).
pub fn init_tracing_with(verbose: bool, tui_sender: Option<ProgressSender>) {
    TRACING_INIT.call_once(|| {
        use tracing_subscriber::prelude::*;

        let filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new(default_filter(verbose)));

        match tui_sender {
            Some(sender) => {
                // TUI-active mode: send tracing into the TUI log pane ONLY.
                // Writing to stderr concurrently with the alternate-screen
                // ratatui draws would paint over the TUI.
                let layer = TuiTracingLayer::new(sender);
                let _ = tracing_subscriber::registry()
                    .with(filter)
                    .with(layer)
                    .try_init();
            }
            None => {
                // Headless / plain mode: keep the fmt-stderr layer.
                let _ = tracing_subscriber::fmt()
                    .with_env_filter(filter)
                    .with_writer(stderr)
                    .with_target(false)
                    .try_init();
            }
        }
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
