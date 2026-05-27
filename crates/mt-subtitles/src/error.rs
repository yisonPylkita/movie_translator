//! Structured error types for subtitle parsing and processing.

use thiserror::Error;

/// Errors produced while parsing ASS/SRT subtitle text.
///
/// Where a line number is reasonably trackable it is included (1-based, as
/// seen in a text editor). `line_no` is `None` when a failure is not tied to a
/// specific source line.
#[derive(Debug, Error)]
pub enum ParseError {
    /// A required field was missing from an event/style/timing line.
    #[error("missing field `{field}`{}", fmt_line(*line_no))]
    MissingField {
        field: String,
        line_no: Option<usize>,
    },

    /// A timestamp could not be parsed.
    #[error("bad time value `{value}`{}", fmt_line(*line_no))]
    BadTime {
        value: String,
        line_no: Option<usize>,
    },

    /// The input was structurally malformed.
    #[error("malformed subtitle: {detail}{}", fmt_line(*line_no))]
    Malformed {
        detail: String,
        line_no: Option<usize>,
    },

    /// An unsupported file extension was supplied to [`crate::load`].
    #[error("unsupported subtitle extension: {0:?}")]
    UnsupportedExtension(Option<String>),

    /// Underlying I/O failure (e.g. reading the file).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

fn fmt_line(line_no: Option<usize>) -> String {
    match line_no {
        Some(n) => format!(" (line {n})"),
        None => String::new(),
    }
}

impl ParseError {
    /// Convenience constructor for a missing field with a line number.
    pub(crate) fn missing_at(field: impl Into<String>, line_no: usize) -> Self {
        ParseError::MissingField {
            field: field.into(),
            line_no: Some(line_no),
        }
    }

    /// Convenience constructor for a malformed-input error with a line number.
    pub(crate) fn malformed_at(detail: impl Into<String>, line_no: usize) -> Self {
        ParseError::Malformed {
            detail: detail.into(),
            line_no: Some(line_no),
        }
    }
}
