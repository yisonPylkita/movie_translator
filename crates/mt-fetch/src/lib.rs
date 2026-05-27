//! Subtitle fetching from remote sources (OpenSubtitles, Podnapisi, NapiProjekt, AnimeSub).

pub mod fetcher;
pub mod providers;
pub mod rate_limiter;
pub mod retry;
pub mod scoring;
pub mod types;

pub use fetcher::SubtitleFetcher;
pub use types::SubtitleMatch;
