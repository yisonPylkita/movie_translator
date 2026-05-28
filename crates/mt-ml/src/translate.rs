//! Translation via the embedded Python `movie_translator.translation`.
//!
//! Calls in-process through PyO3 (see [`crate::backend`]); the underlying
//! `SubtitleTranslator` is cached across calls by a shared `ModelCache` so
//! the model loads once per binary run, not once per file.

use mt_core::{DialogueLine, Result};
use serde::{Deserialize, Serialize};

/// Translation request — kept as a serde-friendly struct purely so callers
/// (mt-pipeline / GPU executor trait) can pass a single value around.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslateRequest {
    /// Dialogue lines to translate.
    pub lines: Vec<DialogueLine>,
    /// Inference device: `"cpu"`, `"mps"`, or `"cuda"`.
    pub device: String,
    /// Translation batch size.
    pub batch_size: u32,
    /// Backend/model name, e.g. `"allegro"`, `"apple"`, `"nllb"`.
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

/// Translate dialogue lines through the embedded Python translator.
///
/// The first call loads the model; subsequent calls with matching
/// `(device, batch_size, model)` reuse the cached instance.
///
/// # Errors
/// Returns [`mt_core::MtError::Parse`] carrying the Python traceback when
/// the in-process translation raises.
pub fn translate(req: &TranslateRequest) -> Result<Vec<DialogueLine>> {
    let proper = req.proper_nouns.as_deref();
    crate::backend::translate(&req.lines, &req.device, req.batch_size, &req.model, proper)
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
            model: "allegro".to_string(),
            proper_nouns: Some(vec!["Luffy".to_string()]),
        };
        let json = serde_json::to_value(&req).expect("serialize");
        assert_eq!(json["device"], "cpu");
        assert_eq!(json["batch_size"], 8);
        assert_eq!(json["model"], "allegro");
        assert_eq!(json["proper_nouns"][0], "Luffy");
        assert_eq!(json["lines"][0]["text"], "Hello");
    }

    #[test]
    fn translate_request_omits_none_proper_nouns() {
        let req = TranslateRequest {
            lines: vec![],
            device: "cpu".to_string(),
            batch_size: 8,
            model: "allegro".to_string(),
            proper_nouns: None,
        };
        let json = serde_json::to_value(&req).expect("serialize");
        assert!(json.get("proper_nouns").is_none());
    }

    /// Real-model end-to-end translation via the embedded Python.
    /// Loads the cached Allegro BiDi model from `./models/allegro`, so it's
    /// `#[ignore]` (heavy: ~2GB model). Run with
    /// `cargo test -p mt-ml --release -- --ignored real_model_translates_via_pyo3`.
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
            model: "allegro".to_string(),
            proper_nouns: None,
        };
        let out = translate(&req).expect("real translate");
        assert_eq!(out.len(), 2);
        // Polish output should be non-empty and differ from the input.
        for (orig, translated) in req.lines.iter().zip(out.iter()) {
            assert!(!translated.text.is_empty(), "translation empty");
            assert_ne!(orig.text, translated.text, "no-op translation");
            eprintln!("EN: {}\nPL: {}\n", orig.text, translated.text);
        }
    }
}
