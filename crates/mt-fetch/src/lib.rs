//! Subtitle fetching from remote sources (OpenSubtitles, Podnapisi, NapiProjekt, AnimeSub).

pub mod align;
pub mod align_ilass;
pub mod fetcher;
pub mod ogladajanime;
pub mod providers;
pub mod rate_limiter;
pub mod retry;
pub mod scoring;
pub mod style_classifier;
pub mod types;
pub mod validator;

pub use align::{
    align_to_reference as align_cross_correlation, detect_op_gap, detect_op_gap_default,
    estimate_offset,
};
pub use align_ilass::align_to_reference as align_ilass;
pub use fetcher::SubtitleFetcher;
pub use style_classifier::classify_styles;
pub use types::SubtitleMatch;
pub use validator::SubtitleValidator;
