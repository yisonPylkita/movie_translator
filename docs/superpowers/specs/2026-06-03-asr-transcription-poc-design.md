# ASR transcription PoC — audio → subtitles feasibility bake-off

**Date:** 2026-06-03
**Status:** Approved (brainstorming) → unattended execution
**Branch:** `poc/asr-transcription`

## Goal

Decide whether (and how) to add audio→subtitle transcription to the translator,
for anime with **English** or **Japanese** voiceover. This is a **proof of
concept / bake-off**, not a production feature: compare ASR engines on real
material, measure quality + speed + integration cost, and produce a report that
names a recommended engine. A later branch productionizes the winner.

## Non-goals

- Wiring ASR into the production pipeline (`crates/mt-pipeline`). Not yet.
- Touching the main project's `.venv` / `pyproject.toml` / `uv.lock`. The PoC is
  fully isolated so it cannot break the production translator or the gate.
- Picking a final UX (CLI flag, stage ordering). The report informs that later.

## Test material

`/Users/w/Downloads/Torrents/.../[Judas] Isekai Ojisan - S01E01.mkv` (a file this
translator already produced). One container, everything needed:

| Track | Type | Lang | Role |
| --- | --- | --- | --- |
| 1 | audio opus | jpn | Japanese original audio (ASR input) |
| 2 | audio opus | eng | English dub audio (ASR input) |
| 3 | subtitle ass | eng | "English (Original)" — **reference transcript** |
| 4 | subtitle srt | pol | Polish (ogladajanime-ocr) — secondary ref |
| 5/6 | subtitle ass | pol | Polish (Allegro / Apple) — secondary ref |

24 min/episode, 13 episodes available. We use a 3-min segment for iteration and
one full episode for the headline numbers.

## Engines (the approaches)

| Engine | Accel on this Mac | Notes |
| --- | --- | --- |
| openai-whisper | PyTorch / MPS | reference baseline |
| faster-whisper | CTranslate2 / CPU | CT2 has no MPS; int8 CPU still fast |
| whisper.cpp | Metal | ggml models; via pywhispercpp or built CLI |
| mlx-whisper | Apple MLX / Metal | Apple-Silicon-native |
| WhisperX | MPS + wav2vec2 | VAD + forced alignment → timestamp quality |
| kotoba-whisper | MPS | Japanese-specialized distil-Whisper (JP only) |
| Apple SpeechAnalyzer | on-device (macOS 26) | Swift bridge like Apple Translation; **timeboxed** |

Whisper-family engines run at `small` and `large-v3`. Each engine lives in its
own isolated venv (`benchmarks/asr/envs/<engine>/`) so mutually-incompatible
pins (ctranslate2 vs torch, pyannote) never collide. Adapters are invoked as
subprocesses in their own env.

## Harness

```
benchmarks/asr/
├── prep_audio.py      Extract tracks 1/2 → 16 kHz mono wav (+ 3-min segment). System ffmpeg.
├── adapters/          One <engine>_adapter.py per engine: stdin/argv -> segments JSON.
│                      Uniform contract: transcribe(wav, lang) -> [{start_ms,end_ms,text}], meta{rtf, model, peak_ram_mb}.
├── run.py             Orchestrator: per engine, ensure venv, run adapter subprocess, collect results/. SERIAL inference.
├── eval.py            Pure metrics: WER/CER (jiwer), chrF (sacrebleu), timing error, segment parsing. Unit-tested.
├── envs/<engine>/     Isolated uv venvs (GITIGNORED).
├── results/<engine>_<lang>_<model>.json   Committed.
├── REPORT.md          Final comparison + recommendation. Committed.
└── README.md          How to run / reproduce.
```

Adapter contract (every engine emits the same shape):

```json
{"engine":"mlx-whisper","model":"large-v3","lang":"ja","wav":"...",
 "segments":[{"start_ms":1200,"end_ms":3400,"text":"..."}],
 "meta":{"rtf":0.12,"audio_s":180.0,"infer_s":21.6,"peak_ram_mb":2100,"ok":true,"error":null}}
```

A failed engine writes `ok:false` + `error` and is reported as such — one bad
install never aborts the bake-off.

## Evaluation — layered

1. **WER / CER** (jiwer): EN-dub transcript (track 2 audio) vs EN subs (track 3).
   *Caveat reported inline:* dubtitles ≠ sub script, so this is an upper bound on
   true error, not a clean WER. Normalized (lowercase, strip punctuation) before
   scoring.
2. **End-to-end usefulness** (the real question): JP audio (track 1) →
   transcribe (ja) → translate JP→EN via the existing translator → **chrF**
   vs EN subs (track 3). Tells us whether ASR-sourced subs are good enough to
   translate from. Also EN-dub → transcribe → chrF vs EN subs (already English).
3. **Timestamp quality:** for lines matched to the reference by text similarity,
   mean |start-error| and |end-error| in ms. Flags engines whose segmentation is
   too coarse/fine for readable subtitles.
4. **Operational:** realtime factor (infer_s / audio_s), model download size,
   peak RAM, runs-on-this-Mac (yes/no), JP support, word-vs-segment timestamps,
   integration effort estimate.

## Scenarios

`{EN dub, JP orig}` × `{3-min segment, 1 full episode}` × engines ×
`{small, large-v3}` (whisper-family). Qualitative notes captured for OP/ED
singing, the narrator voice, and overlapping speech.

## Parallelism (respects the one-GPU rule)

- **Parallel** (no GPU contention): per-engine venv creation + install + model
  download (disjoint dirs), audio extraction, engine-docs research, drafting
  report sections.
- **Serial**: all actual transcription inference (Metal/MPS/CPU all contend) and
  the JP→EN translate step (the production GPU worker). Run one at a time.

## Risks / honesty

- **Dependency hell** (whisperx/pyannote/ctranslate2 vs torch 2.12, transformers
  5.0rc): mitigated by per-engine isolated venvs. An engine that can't install is
  reported "failed to install: <reason>", not silently dropped.
- **Apple SpeechAnalyzer** Swift bridge may not build (macOS version, SDK).
  Timeboxed; reported as effort/feasibility if it doesn't run.
- **No clean JP ground truth**: JP accuracy is judged end-to-end (layer 2) +
  manual/LLM spot-check, never claimed as measured WER. The report states clearly
  what is measured vs estimated.

## Deliverable

`benchmarks/asr/REPORT.md` + committed `results/*.json`, and a summary back to
the user: comparison table across all engines (EN + JP), with a recommendation
(which engine for EN, which for JP) and the speed/quality/integration tradeoffs.
