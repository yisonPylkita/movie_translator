# Audio→subtitle transcription — production integration

**Date:** 2026-06-04 · **Branch:** `feat/asr-transcription` · **Status:** approved (PoC follow-up)

Integrate the two bake-off winners (see `benchmarks/asr/REPORT.md`) into the
pipeline behind CLI flags: **Apple SpeechAnalyzer** (primary) and
**mlx-whisper large-v3** (fallback / non-Apple-26).

## UX

`movie-translator <input> --transcribe [--transcribe-engine {apple,whisper}]`
(default `apple`). When enabled, the pipeline sources English dialogue from the
**English audio track via ASR** when no English subtitle text is found
(embedded/fetched/reference/burned-in all missed). Without the flag, behavior is
unchanged. Translation-only consumer: we transcribe the **eng** audio track
(the EN→PL translator can't consume Japanese), so JA transcription stays
PoC-only for now.

## Python (`movie_translator/transcription/`)

- `transcribe_to_srt(video_path, output_dir, language='en', engine='apple') -> Path | None`
  — probe the video for an audio track in `language`; extract 16 kHz mono wav;
  run the engine; post-process; write `transcribed_eng.srt`. `None` (skip, not
  fail) when: no such audio track, engine unavailable, or no usable segments.
- `apple_backend.py` — macOS 26 SpeechAnalyzer via a Swift bridge
  (`swift/transcribe_bridge.swift`, compiled on first use, mirroring the Apple
  Translation bridge). Coarse utterances → `splitter.py`.
- `whisper_backend.py` — mlx-whisper `large-v3` (Metal). Output →
  `postfilter.py` (Whisper hallucinates on ED music/silence; see PoC).
- `splitter.py` (unit-tested): split multi-sentence segments on `.!?。！？` with
  duration allocated proportionally to text length.
- `postfilter.py` (unit-tested): drop empties, drop segments starting past
  audio end, clamp ends, collapse consecutive duplicate texts (the
  "ご視聴ありがとうございました" loop).
- New dep: `mlx-whisper` (darwin/arm64 marker) in the MAIN venv; model pulled
  from HF on first use (mlx-community/whisper-large-v3-mlx).

## Rust

- `mt-core`: `PipelineConfig.enable_transcription: bool`, `transcribe_engine: String`.
- `mt-ml`: import `movie_translator.transcription` into `Modules`;
  `transcribe_to_srt(video, out_dir, lang, engine) -> Result<Option<PathBuf>>`.
- `mt-pipeline`: `GpuExecutor::transcribe(...)` + `Job::Transcribe` through the
  serialized worker (Apple runs on ANE but still routes through the worker —
  one ML lane, simpler invariants). New `stages/transcribe.rs`; orchestrator
  runs it after extract_english + pending-OCR, only when
  `english_source.is_none() && config.enable_transcription`, before the
  `NoEnglishSource` bail.
- `mt-cli`: `--transcribe`, `--transcribe-engine` (clap value_parser).

## Tests

Python: splitter + postfilter unit tests (TDD); availability gating. Rust:
stage skip-conditions unit test; real-ASR runs stay `#[ignore]` per repo
convention. Gate: `just check && just test && just py-test`.

## Demo (end-to-end proof)

Strip subtitle tracks from one Isekai Ojisan episode (`-map 0:v -map 0:a -c
copy -sn`) so the pipeline has no text source, then
`just run <file> --no-fetch --transcribe` → EN dub audio → ASR → translate →
Polish track muxed. Verify the output file carries the new PL track.
