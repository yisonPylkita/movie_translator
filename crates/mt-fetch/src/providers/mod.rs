//! Subtitle provider implementations.

pub mod animesub;
pub mod napiprojekt;
pub mod opensubtitles;
pub mod podnapisi;

use std::path::Path;

use crate::types::SubtitleMatch;
use crate::retry::FetchError;
use mt_core::MediaIdentity;

/// Trait for subtitle providers (mirrors Python `SubtitleProvider` Protocol).
pub trait SubtitleProvider: Send + Sync {
    fn name(&self) -> &str;

    fn search(
        &self,
        identity: &MediaIdentity,
        languages: &[&str],
    ) -> Result<Vec<SubtitleMatch>, FetchError>;

    fn download(
        &self,
        match_: &SubtitleMatch,
        output_path: &Path,
    ) -> Result<(), FetchError>;
}
