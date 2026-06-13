//! Translation dispatch: Apple (Rust-native) or MLX (PyO3/embedded Python).
//!
//! The Apple Translation backend runs entirely in Rust (calls the Swift bridge
//! binary, handles sentence merging + enhancements natively).  The MLX backend
//! (Allegro BiDi via MLX) stays in Python and is called through PyO3.

use mt_core::{DialogueLine, Result};
use serde::{Deserialize, Serialize};
use tracing::info;

/// Translation request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslateRequest {
    /// Dialogue lines to translate.
    pub lines: Vec<DialogueLine>,
    /// Inference device: `"cpu"`, `"mps"`, or `"cuda"`.
    pub device: String,
    /// Translation batch size.
    pub batch_size: u32,
    /// Backend/model name: `"apple"` or `"mlx"`.
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

/// Translate dialogue lines.
///
/// Dispatches to the appropriate backend:
/// - `"apple"` → Rust-native Apple Translation framework via Swift bridge
/// - `"mlx"`   → PyO3 embedded Python (MLX Allegro BiDi model)
///
/// # Errors
/// Returns [`mt_core::MtError`] on failure.
pub fn translate(req: &TranslateRequest) -> Result<Vec<DialogueLine>> {
    match req.model.as_str() {
        "apple" => {
            info!("Using Apple Translation backend (Rust-native)");
            let proper = req.proper_nouns.as_deref();
            crate::apple_translate::translate(&req.lines, req.batch_size, proper)
        }
        _ => {
            info!("Using MLX backend via Python (PyO3)");
            let proper = req.proper_nouns.as_deref();
            crate::backend::translate(&req.lines, &req.device, req.batch_size, &req.model, proper)
        }
    }
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
        // Verify apple routes correctly (it won't actually translate on CI
        // without the Swift bridge, but the dispatch logic is correct).
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
        // This will error on non-macOS or without Swift, but we check the
        // error comes from apple_translate, not from pyo3.
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

    /// Real-model end-to-end translation via PyO3 (MLX).
    /// Loads the cached Allegro BiDi model from `./models/allegro`, so it's
    /// `#[ignore]` (heavy: ~2GB model).
    #[test]
    #[ignore]
    fn real_model_translates_via_pyo3() {
        let req = TranslateRequest {
            lines: vec![
                DialogueLine {
                    start_ms: 0,
                    end_ms: 1000,
                    text: "Hello, how are you?".to_string(),
                },
                DialogueLine {
                    start_ms: 1000,
                    end_ms: 2000,
                    text: "I am fine, thank you.".to_string(),
                },
            ],
            device: "cpu".to_string(),
            batch_size: 2,
            model: "mlx".to_string(),
            proper_nouns: None,
        };
        let out = translate(&req).expect("real translate");
        assert_eq!(out.len(), 2);
        for (orig, translated) in req.lines.iter().zip(out.iter()) {
            assert!(!translated.text.is_empty(), "translation empty");
            assert_ne!(orig.text, translated.text, "no-op translation");
            eprintln!("EN: {}\nPL: {}\n", orig.text, translated.text);
        }
    }
}
