//! Translation via the Python `ml/translate.py` helper script.
//!
//! Mirrors the `TranslateTask` GPU task in `movie_translator/gpu_queue.py`.

use crate::runner::run_script_json;
use mt_core::{DialogueLine, Result};
use serde::{Deserialize, Serialize};

/// JSON request sent to `ml/translate.py` on stdin.
///
/// Fields mirror the script contract (and the `TranslateTask` dataclass).
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

/// JSON response from `ml/translate.py` on stdout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslateResponse {
    /// Translated dialogue lines (same count/order as the request).
    pub lines: Vec<DialogueLine>,
}

/// Translate dialogue lines by spawning `ml/translate.py`.
///
/// The Python side loads the model once per call, translates, and returns the
/// translated [`DialogueLine`]s in the same order.
///
/// # Errors
/// Propagates [`mt_core::MtError::Subprocess`] (with stderr) on script failure.
pub fn translate(req: &TranslateRequest) -> Result<Vec<DialogueLine>> {
    translate_with_args(req, &[])
}

/// Like [`translate`], but forwards extra CLI args to the script (e.g.
/// `--self-test`). Internal/testing helper.
pub(crate) fn translate_with_args(
    req: &TranslateRequest,
    args: &[&str],
) -> Result<Vec<DialogueLine>> {
    let resp: TranslateResponse = run_script_json("translate.py", args, Some(req))?;
    Ok(resp.lines)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unit test: request serde shape matches the script contract.
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

    /// Unit test: `proper_nouns: None` is omitted from the serialized request.
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

    /// Unit test: response deserialises from the script's stdout shape.
    #[test]
    fn translate_response_deserialize_shape() {
        let json = r#"{"lines":[{"start_ms":0,"end_ms":1000,"text":"[xl] Hello"}]}"#;
        let resp: TranslateResponse = serde_json::from_str(json).expect("deserialize");
        assert_eq!(resp.lines.len(), 1);
        assert_eq!(resp.lines[0].text, "[xl] Hello");
    }

    /// Integration test: drive `translate` via `ml/translate.py --self-test`,
    /// exercising the real spawn + stdin/stdout JSON path (no model needed).
    /// Not `#[ignore]` — runs in CI.
    #[test]
    fn translate_self_test_via_public_path() {
        let req = TranslateRequest {
            lines: vec![DialogueLine {
                start_ms: 0,
                end_ms: 1000,
                text: "Hello".to_string(),
            }],
            device: "cpu".to_string(),
            batch_size: 8,
            model: "allegro".to_string(),
            proper_nouns: None,
        };
        let lines = translate_with_args(&req, &["--self-test"]).expect("self-test translate");
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].text, "[xl] Hello");
    }
}
