//! Integration tests exercising the Rust runner + JSON contract against the
//! Python `ml/*.py` scripts' `--self-test` paths.
//!
//! These spawn the real Python scripts (via `uv run python`) but pass
//! `--self-test`, so NO ML models / torch / cv2 are needed. They run in CI to
//! verify the spawn + stdin/stdout-JSON plumbing end-to-end through the
//! public `run_script_json` runner.
//!
//! (The thin typed wrappers `translate` / `ocr_*` / `inpaint` are covered by
//! self-test integration tests inside each module via the `*_with_args`
//! crate-internal helpers.)

use mt_core::{BoundingBox, BurnedInResult, DialogueLine, OCRResult};
use mt_ml::run_script_json;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Serialize)]
struct TranslateReq {
    lines: Vec<DialogueLine>,
    device: String,
    batch_size: u32,
    model: String,
    proper_nouns: Option<Vec<String>>,
}

#[derive(Deserialize)]
struct TranslateResp {
    lines: Vec<DialogueLine>,
}

/// translate.py --self-test prefixes each line's text with "[xl] ".
#[test]
fn translate_self_test_roundtrip() {
    let req = TranslateReq {
        lines: vec![
            DialogueLine {
                start_ms: 0,
                end_ms: 1000,
                text: "Hello".to_string(),
            },
            DialogueLine {
                start_ms: 1000,
                end_ms: 2000,
                text: "World".to_string(),
            },
        ],
        device: "cpu".to_string(),
        batch_size: 8,
        model: "allegro".to_string(),
        proper_nouns: None,
    };

    let resp: TranslateResp =
        run_script_json("translate.py", &["--self-test"], Some(&req)).expect("translate self-test");

    assert_eq!(resp.lines.len(), 2);
    assert_eq!(resp.lines[0].text, "[xl] Hello");
    assert_eq!(resp.lines[1].text, "[xl] World");
    assert_eq!(resp.lines[0].start_ms, 0);
    assert_eq!(resp.lines[1].end_ms, 2000);
}

#[derive(Deserialize)]
struct PgsResp {
    srt_path: Option<PathBuf>,
}

/// ocr.py --type pgs --self-test echoes "<video>.srt".
#[test]
fn ocr_pgs_self_test_roundtrip() {
    let resp: PgsResp = run_script_json::<(), _>(
        "ocr.py",
        &[
            "--type",
            "pgs",
            "--video",
            "/tmp/foo.mkv",
            "--track-index",
            "2",
            "--work-dir",
            "/tmp/wd",
            "--self-test",
        ],
        None,
    )
    .expect("ocr pgs self-test");

    assert_eq!(resp.srt_path.as_deref(), Some(std::path::Path::new("/tmp/foo.mkv.srt")));
}

/// ocr.py --type burned_in --self-test returns a fixed BurnedInResult.
#[test]
fn ocr_burned_in_self_test_roundtrip() {
    let resp: BurnedInResult = run_script_json::<(), _>(
        "ocr.py",
        &[
            "--type",
            "burned_in",
            "--video",
            "/tmp/foo.mkv",
            "--output-dir",
            "/tmp/out",
            "--crop-ratio",
            "0.3",
            "--fps",
            "2",
            "--self-test",
        ],
        None,
    )
    .expect("ocr burned_in self-test");

    assert_eq!(resp.srt_path.to_str(), Some("/tmp/foo.mkv.srt"));
    assert_eq!(resp.ocr_results.len(), 1);
    assert_eq!(resp.ocr_results[0].text, "self-test subtitle");
    assert_eq!(resp.ocr_results[0].boxes[0].width, 0.5);
}

#[derive(Deserialize)]
struct InpaintResp {
    output_path: PathBuf,
}

/// inpaint.py --self-test copies the input to the output and echoes the path.
#[test]
fn inpaint_self_test_roundtrip() {
    // Build an input video and an ocr-results JSON file the script can read.
    let dir = tempfile::tempdir().expect("tempdir");
    let video = dir.path().join("in.mp4");
    let output = dir.path().join("out.mp4");
    std::fs::write(&video, b"dummy video bytes").expect("write video");

    let ocr_results = vec![OCRResult {
        timestamp_ms: 1000,
        text: "x".to_string(),
        boxes: vec![BoundingBox {
            x: 0.1,
            y: 0.8,
            width: 0.5,
            height: 0.1,
        }],
    }];
    let ocr_path = dir.path().join("ocr.json");
    std::fs::write(&ocr_path, serde_json::to_vec(&ocr_results).unwrap()).expect("write ocr json");

    let resp: InpaintResp = run_script_json::<(), _>(
        "inpaint.py",
        &[
            "--video",
            video.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--device",
            "cpu",
            "--backend",
            "lama",
            "--ocr-results",
            ocr_path.to_str().unwrap(),
            "--self-test",
        ],
        None,
    )
    .expect("inpaint self-test");

    assert_eq!(resp.output_path, output);
    // The self-test copies input -> output.
    let copied = std::fs::read(&output).expect("read output");
    assert_eq!(copied, b"dummy video bytes");
}

/// #[ignore] integration test: real translation (loads a model). Present but
/// ignored so CI does not need torch. Run with:
///   cargo test -p mt-ml -- --ignored
#[test]
#[ignore]
fn translate_real_model() {
    use mt_ml::{translate, TranslateRequest};

    let req = TranslateRequest {
        lines: vec![DialogueLine {
            start_ms: 0,
            end_ms: 1000,
            text: "Cześć".to_string(),
        }],
        device: "cpu".to_string(),
        batch_size: 8,
        model: "allegro".to_string(),
        proper_nouns: None,
    };
    let lines = translate(&req).expect("real translate");
    assert_eq!(lines.len(), 1);
    assert!(!lines[0].text.is_empty());
}
