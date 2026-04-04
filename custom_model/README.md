# Custom EN→PL Subtitle Translation Model

Purpose-built English-to-Polish translation model for anime and movie subtitles. Single language pair, context-aware architecture, trained on LLM-generated + human-reviewed parallel data.

## Status

Design phase. No training data or model code yet.

## Directory structure

```
custom_model/
  docs/
    design.md                          # Architecture and dataset design document
    translation-quality-experiments.md # Research: Allegro vs Apple backend experiments
  scripts/
    translation_experiments.py         # Experiment harness for backend comparison
  dataset/                             # (future) parallel EN-PL corpus
  training/                            # (future) training scripts and exports
```

## Key design decisions

- **Single pair only**: EN→PL. Every parameter is dedicated to this one language pair. No multilingual waste.
- **200-250M parameters**: matches current Allegro footprint, fits Apple Silicon with INT8 quantization (~200MB).
- **Context-aware architecture**: separate context encoder (show/episode metadata) + dialogue encoder (sliding window of surrounding lines) + translation decoder.
- **Dataset decoupled from model**: JSON files as source of truth, JSONL export for training. The dataset is a long-lived asset independent of model architecture.

See [docs/design.md](docs/design.md) for the full design document.
