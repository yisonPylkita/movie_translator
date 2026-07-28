---
name: ml-stage-debug
description: Debug translation, OCR, or inpainting stage failures. All stages are pure Rust (no Python). Localizes the failing module; the parent agent fixes.
---

# ML Stage Debug

All ML stages in this project run in pure Rust — zero Python
dependencies. Stages are called through the serialised GPU worker
(`crates/mt-pipeline/src/worker.rs`).

## Translation (`crates/mt-ml/src/translate/apple.rs`)

Apple Translation framework via compiled Swift bridge. macOS 26+.
Sentence merging in `crates/mt-subtitles/src/sentence_merger.rs`.

## OCR (`crates/mt-ml/src/ocr/vision.rs`)

Apple Vision via compiled Swift bridge.
PGS parsing in `crates/mt-media/src/pgs_parser.rs`. macOS only.

## Inpainting (`crates/mt-ml/src/inpaint.rs`)

Pure Rust Telea algorithm. Works on all platforms.

## Transcription (`crates/mt-ml/src/transcription.rs`)

Apple SpeechAnalyzer (Swift bridge) or whisper-cli subprocess.
VAD is energy-based (inline).

## Diagnostic flow

1. Compile-time failure → fix the Rust code.
2. Runtime failure → check Swift bridge compilation output, ffmpeg
   stderr (`RUST_LOG=debug`), or system dependency (whisper-cli path,
   macOS version).

## What you don't do

- Don't add Python back.
- Don't push changes; localise and hand back to the parent agent.
