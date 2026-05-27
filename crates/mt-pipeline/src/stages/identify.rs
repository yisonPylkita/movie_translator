//! Identify stage — extract media identity from the video file.
//!
//! Port of `movie_translator/stages/identify.py`.

use mt_core::PipelineContext;
use mt_discovery::identify_media;

use crate::error::Result;

/// Stage role name (matches the Python `IdentifyStage.name`).
pub const NAME: &str = "identify";

/// Identify the media and store it on the context.
///
/// Port of `IdentifyStage.run`. The Python version also emits metrics spans
/// for the identity fields; metrics are a Python-only concern here.
pub fn run(mut ctx: PipelineContext) -> Result<PipelineContext> {
    tracing::info!(
        "Identifying: {}",
        ctx.video_path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default()
    );
    ctx.identity = Some(identify_media(&ctx.video_path)?);
    Ok(ctx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_matches_python() {
        assert_eq!(NAME, "identify");
    }
}
