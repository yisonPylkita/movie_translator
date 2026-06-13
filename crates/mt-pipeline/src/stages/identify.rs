//! Identify stage — extract media identity from the video file.

use mt_core::PipelineContext;
use mt_discovery::identify_media;

use crate::error::Result;
use tracing::info;

/// Stage role name.
pub const NAME: &str = "identify";

/// Identify the media and store it on the context.
pub fn run(mut ctx: PipelineContext) -> Result<PipelineContext> {
    info!(
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
    fn name() {
        assert_eq!(NAME, "identify");
    }
}
