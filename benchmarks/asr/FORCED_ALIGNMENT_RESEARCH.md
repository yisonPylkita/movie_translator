# Forced alignment research — improving ASR subtitle timing

**Date:** 2026-06-13 · **Status:** Research complete, not implemented

## The problem

Current ASR output has poor timing granularity:

| Engine | Timing error | Root cause |
| --- | --- | --- |
| Apple SpeechAnalyzer | ~2 s (boundary error) | Coarse utterances → proportional character-length split |
| mlx-whisper large-v3 | ~1.2 s (jitter) | Per-segment timestamps from the Whisper decoder |

Subtitle-grade timing needs ~100 ms accuracy so lines pop in/out on screen at the right moment, especially for fast dialogue.

## Three approaches compared

### 1. WhisperX forced alignment (wav2vec2 CTC)

**How it works:** Transcribe with any Whisper backend, then re-align the transcript to the audio using a wav2vec2 CTC model. The alignment model predicts character-level timestamps by running the audio through a pre-trained wav2vec2 and beam-searching the best time alignment for the known text.

**Dependencies:**
- `torchaudio` built-in `WAV2VEC2_ASR_BASE_960H` (95 MB model, already available at the project level)
- whisperx package (`alignment` submodule) — ~3.7 MB + its own wav2vec2 wrapper

**Accuracy:** Best in class — **55 ms timing error** measured in the bake-off (see `REPORT.md`).

**Language support:** EN only (the torchaudio model is English-finetuned). A Japanese wav2vec2 exists (`jonatasgrosman/wav2vec2-large-xlsr-53-japanese`, 1.2 GB) but the bake-off found **WhisperX alignment breaks on Japanese** (collapses to 6–9 segments with 10 s+ timing errors).

**Integration cost:**
```
from whisperx.alignment import load_align_model, align

model, metadata = load_align_model(language_code="en", device="mps")
result = align(transcript_segments, model, metadata, audio_waveform, device="mps")
```

Can run as a post-processing step on *any* ASR output (mlx-whisper or Apple), no need to re-transcribe. The alignment step takes ~1–2 s per minute of audio on MPS.

**Our-specific concerns:**
- The `whisperx.alignment` module has many transitive deps (`torch-audiomentations`, `torch-pitch-shift`, `torchmetrics`, etc.) — about 15 extra packages
- We already have `torch` and `transformers`, so it's not *that* bad
- The EN-only limitation is fine — the user confirmed we only transcribe EN
- wav2vec2 alignment produces word-level timestamps; we'd need to merge words back into subtitle lines

### 2. Stable-ts (stable-whisper)

**How it works:** Modifies the Whisper decoding loop to produce more stable timestamps by suppressing silence hallucination and refining cross-attention weights. Doesn't need a separate alignment model.

**Dependencies:**
- `stable-ts` on PyPI — lightweight, wraps openai-whisper or mlx-whisper
- Pulls in `openai-whisper` as a dependency (which we don't otherwise use)

**Accuracy:** Better than vanilla Whisper but not as good as wav2vec2 alignment. Rough estimate: ~200–500 ms timing error. No bake-off data for this specific project, but the WhisperX 55 ms is documented.

**Language support:** Works with any language Whisper supports (EN + JA).

**Integration cost:**
```
import stable_whisper
model = stable_whisper.load_model('large-v3')
result = model.transcribe(audio_path, suppress_silence=False)
# result contains word-level timestamps
```

**Our-specific concerns:**
- Main selling point is that it works on JA too — but we're EN-only
- Adds `openai-whisper` as a dep (already have `mlx-whisper`)
- Timing improvement is incremental, not transformative

### 3. Direct CTC alignment (torchaudio + transformers, no whisperx)

**How it works:** Same wav2vec2 alignment as WhisperX but implemented directly with `torchaudio` and `transformers`, avoiding the whisperx package entirely.

**Dependencies:** Zero new deps — `torchaudio` and `transformers` are already in `pyproject.toml`.

**Accuracy:** Identical to WhisperX's alignment (same underlying model).

**Language support:** EN only (torchaudio built-in model).

**Integration cost:** More code to write (about ~50 lines for the alignment loop). No extra packages.

Example sketch:
```python
import torch
import torchaudio
import torchaudio.functional as F

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)
labels = bundle.get_labels()
dictionary = {c.lower(): i for i, c in enumerate(labels)}

# Run CTC beam search on audio -> character probabilities
waveform, sr = torchaudio.load(audio_path)
with torch.inference_mode():
    emissions, _ = model(waveform)

# Align known transcript to emission timesteps
tokens = [dictionary.get(c, 0) for c in transcript.lower()]
alignment = F.forced_align(emissions[0], torch.tensor([tokens]), blank=0)
# alignment gives per-character timestamps (as emission frame indices)
```

**Our-specific concerns:**
- Most lean approach (zero new deps)
- Requires implementing the segment-merging logic ourselves
- Works only on English, which is fine
- Well-documented in torchaudio examples

## Recommendation

**Use approach 3 (direct CTC alignment via torchaudio) when timing polish is desired.**

Rationale:
1. ⭐ **Zero new dependencies** — `torchaudio` and `transformers` are already in the project
2. 🏆 **Best timing** — identical to WhisperX's 55 ms accuracy
3. 🪶 **Lightest integration** — ~50 lines of Python, no pip fire
4. 🎯 **EN-only** — matches the project's transcription scope
5. ⏱️ **Post-hoc step** — runs on *any* existing ASR output (Apple or Whisper), no re-transcription needed

Trade-off vs WhisperX: more code to write, but avoids adding 15 extra packages with heavy deps (`torch-audiomentations` → `audiomentations`, `torch-pitch-shift`, etc.).

## When to implement

- **Now:** Not critical — the current proportional-split timing is adequate for a fallback feature
- **Trigger:** If users report "subtitles feel off-sync" or if timing becomes a quality bottleneck in the EN→PL translation pipeline
- **Implementation cost:** ~0.5 days for the alignment module + tests + TUI progress

## References

- Bake-off data: `benchmarks/asr/REPORT.md`
- torchaudio forced alignment tutorial: https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html
- WhisperX source (for reference): https://github.com/m-bain/whisperX
