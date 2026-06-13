use thiserror::Error;

/// Top-level error type for the `mt-*` crate family.
#[derive(Debug, Error)]
pub enum MtError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("parse error: {0}")]
    Parse(String),

    #[error("subprocess `{cmd}` failed (exit {code:?}): {stderr}")]
    Subprocess {
        cmd: String,
        code: Option<i32>,
        stderr: String,
    },

    #[error("network error: {0}")]
    Network(String),

    #[error("path resolution failed: {0}")]
    PathResolution(String),
}

/// Convenience `Result` alias for this crate.
pub type Result<T> = std::result::Result<T, MtError>;

#[cfg(test)]
mod tests {
    use std::io::{Error, ErrorKind};

    use super::*;

    #[test]
    fn parse_error_display() {
        let e = MtError::Parse("bad timestamp".to_string());
        assert!(e.to_string().contains("bad timestamp"));
    }

    #[test]
    fn subprocess_error_display() {
        let e = MtError::Subprocess {
            cmd: "ffmpeg".to_string(),
            code: Some(1),
            stderr: "no such file".to_string(),
        };
        let msg = e.to_string();
        assert!(msg.contains("ffmpeg"));
        assert!(msg.contains("no such file"));
    }

    #[test]
    fn io_error_from_conversion() {
        let io_err = Error::new(ErrorKind::NotFound, "file missing");
        let mt_err = MtError::from(io_err);
        assert!(matches!(mt_err, MtError::Io(_)));
    }
}
