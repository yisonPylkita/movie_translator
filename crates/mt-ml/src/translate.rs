//! Translation backend — Apple Translation framework (Rust-native).
//!
//! The only supported backend is the Apple Translation framework on macOS,
//! called via a compiled Swift bridge binary.  The MLX/PyO3 backend has been
//! removed — no Python dependency needed.
//!
//! Sentence merging, placeholder protection, and postprocessing are handled
//! in Rust (see `mt-subtitles::sentence_merger` and `mt-subtitles::enhancements`).

use mt_core::{DialogueLine, Result};
use serde::{Deserialize, Serialize};
use tracing::info;

/// Translation request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslateRequest {
    /// Dialogue lines to translate.
    pub lines: Vec<DialogueLine>,
    /// Inference device (unused with Apple backend, kept for compatibility).
    pub device: String,
    /// Translation batch size.
    pub batch_size: u32,
    /// Backend/model name: only `"apple"` is supported.
    pub model: String,
    /// Character names to protect from translation, or `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proper_nouns: Option<Vec<String>>,
}

/// Response from a translate call (translated dialogue lines in input order).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslateResponse {
    pub lines: Vec<DialogueLine>,
}

/// Translate dialogue lines using the Apple Translation framework.
///
/// This is the only backend — the MLX/Python backend has been removed.
/// The Apple backend calls a compiled Swift bridge binary and handles
/// sentence merging + enhancements natively in Rust.
///
/// # Errors
/// Returns [`mt_core::MtError`] if the Apple Translation framework is
/// unavailable or fails.
pub fn translate(req: &TranslateRequest) -> Result<Vec<DialogueLine>> {
    info!(
        "Using Apple Translation backend (batch_size={}, {} lines)",
        req.batch_size,
        req.lines.len()
    );
    let proper = req.proper_nouns.as_deref();
    crate::apple_translate::translate(&req.lines, req.batch_size, proper)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translate_request_serialize_shape() {
        let req = TranslateRequest {
            lines: vec![DialogueLine {
                start_ms: 0,
                end_ms: 1000,
                text: "Hello".to_string(),
            }],
            device: "cpu".to_string(),
            batch_size: 8,
            model: "apple".to_string(),
            proper_nouns: Some(vec!["Luffy".to_string()]),
        };
        let json = serde_json::to_value(&req).expect("serialize");
        assert_eq!(json["device"], "cpu");
        assert_eq!(json["batch_size"], 8);
        assert_eq!(json["model"], "apple");
        assert_eq!(json["proper_nouns"][0], "Luffy");
        assert_eq!(json["lines"][0]["text"], "Hello");
    }

    #[test]
    fn translate_request_omits_none_proper_nouns() {
        let req = TranslateRequest {
            lines: vec![],
            device: "cpu".to_string(),
            batch_size: 8,
            model: "apple".to_string(),
            proper_nouns: None,
        };
        let json = serde_json::to_value(&req).expect("serialize");
        assert!(json.get("proper_nouns").is_none());
    }

    #[test]
    fn dispatch_routes_apple() {
        let req = TranslateRequest {
            lines: vec![DialogueLine {
                start_ms: 0,
                end_ms: 1000,
                text: "Hello".to_string(),
            }],
            device: "cpu".to_string(),
            batch_size: 4,
            model: "apple".to_string(),
            proper_nouns: None,
        };
        let result = translate(&req);
        if let Err(e) = &result {
            let msg = e.to_string();
            // Should reference Swift-related issues, not Python
            assert!(
                msg.contains("Swift") || msg.contains("swiftc") || msg.contains("bridge"),
                "Apple backend error should mention Swift: {msg}"
            );
        }
    }
}
