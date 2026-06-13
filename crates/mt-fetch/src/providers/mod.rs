//! Subtitle provider implementations.

pub mod animesub;
pub mod napiprojekt;
pub mod opensubtitles;
pub mod podnapisi;

use std::path::Path;

use mt_core::MediaIdentity;
use reqwest::blocking::Client;
use tracing::warn;

use crate::retry::FetchError;
use crate::types::SubtitleMatch;

/// Build a blocking reqwest client with the given user agent.
///
/// If the configured builder fails (e.g. transient TLS backend init failure),
/// fall back to a bare `Client::new()` rather than panicking at construction
/// time. Provider constructors are infallible, so a build failure must not
/// abort the whole process.
pub(crate) fn build_blocking_client(user_agent: &str) -> Client {
    Client::builder()
        .user_agent(user_agent)
        .build()
        .unwrap_or_else(|e| {
            warn!("failed to build configured HTTP client ({e}); using default client");
            Client::new()
        })
}

/// Trait for subtitle providers.
pub trait SubtitleProvider: Send + Sync {
    fn name(&self) -> &str;

    fn search(
        &self,
        identity: &MediaIdentity,
        languages: &[&str],
    ) -> Result<Vec<SubtitleMatch>, FetchError>;

    fn download(&self, match_: &SubtitleMatch, output_path: &Path) -> Result<(), FetchError>;
}
