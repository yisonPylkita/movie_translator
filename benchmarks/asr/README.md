# ASR transcription bake-off (PoC)

Feasibility study: can we add audio→subtitle transcription to the translator,
for anime with English or Japanese voiceover? This compares ASR engines on a
real dual-audio episode and reports a recommendation. **Not wired into the
production pipeline** — see `docs/superpowers/specs/2026-06-03-asr-transcription-poc-design.md`.

## Layout

```
prep_audio.py    Extract JP (track 1) + EN-dub (track 2) audio -> 16 kHz mono wav
                 (+ 180 s iteration segment) and EN/PL reference subs. System ffmpeg.
adapters/        One <engine>_adapter.py per engine, each run in its OWN venv.
                 Uniform contract (argv: wav lang model out.json) via _common.py.
run.py           Orchestrator. Runs adapters serially (one-GPU rule), resumable.
eval.py          Pure scoring helpers (WER/CER/chrF/timing). Unit-tested (test_eval.py).
score.py         Scores results/*.json vs the reference subs -> scores_<variant>.json.
envs/<engine>/   Isolated uv venvs (GITIGNORED — rebuild with the commands below).
audio/           Extracted wavs (GITIGNORED — regenerate with prep_audio.py).
refs/            EN + PL reference subtitle tracks (committed).
results/         Per-config result JSON + scores (committed).
REPORT.md        Findings + recommendation.
```

## Reproduce

```bash
# 1. isolated per-engine venvs (Python 3.12; whisperx on 3.11)
for e in openai faster mlx whispercpp kotoba eval; do uv venv benchmarks/asr/envs/$e --python 3.12; done
uv venv benchmarks/asr/envs/whisperx --python 3.11
uv pip install --python benchmarks/asr/envs/openai/bin/python     openai-whisper
uv pip install --python benchmarks/asr/envs/faster/bin/python     faster-whisper
uv pip install --python benchmarks/asr/envs/mlx/bin/python        mlx-whisper
uv pip install --python benchmarks/asr/envs/whisperx/bin/python   whisperx
uv pip install --python benchmarks/asr/envs/whispercpp/bin/python pywhispercpp
uv pip install --python benchmarks/asr/envs/kotoba/bin/python     transformers torch accelerate librosa soundfile
uv pip install --python benchmarks/asr/envs/eval/bin/python       jiwer sacrebleu pysubs2 pytest

# 2. prep audio + refs (defaults to the Isekai Ojisan S01E01 dual-audio mkv)
python3 benchmarks/asr/prep_audio.py [/path/to/episode.mkv]

# 3. run the matrix (serial, resumable) + score
python3 benchmarks/asr/run.py --variant seg
benchmarks/asr/envs/eval/bin/python benchmarks/asr/score.py --variant seg
```

## Engines

| Engine | Accel (this Mac) | Notes |
| --- | --- | --- |
| openai-whisper | MPS (fp16 off) | reference baseline |
| faster-whisper | CPU int8 | CTranslate2; no MPS |
| mlx-whisper | MLX / Metal | Apple-Silicon-native |
| whisper.cpp | Metal | pywhispercpp, ggml |
| WhisperX | CPU int8 + align | VAD + wav2vec2 forced alignment |
| kotoba-whisper | MPS | JP-specialized; JP only |
