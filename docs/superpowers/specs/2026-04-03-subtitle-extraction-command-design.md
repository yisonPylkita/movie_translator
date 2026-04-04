# Subtitle Extraction Command Design

**Date:** 2026-04-03

## Problem

Videos (especially anime) sometimes exist in two versions: a Polish version with good subtitle translations but poor video quality, and an English/Japanese version with high video quality. We need a way to extract Polish (or English) subtitles from one video and apply them to another.

Subtitles may be:
- **Embedded text tracks** (ASS, SRT) — trivial to extract
- **Burned-in** (hardcoded into video frames) — requires OCR

## Solution

Two changes to the CLI:

1. **New `extract` subcommand** — extracts subtitles from videos (text tracks + OCR for burned-in), outputs SRT files + JSON manifest.
2. **New `--external-subs` flag** on the default (translate) command — points to a directory of pre-extracted subtitles that get added as additional tracks during muxing.

## CLI Structure

```
movie-translator <input> [translate-options]        # existing behavior (translate)
movie-translator extract <input> [extract-options]  # new extraction command
```

The bare command (no subcommand) remains the translate flow — no breaking changes.

### Extract subcommand

```
movie-translator extract <input> --output <dir> --ocr-language <lang> [-v]
```

- `<input>` — video file or directory
- `--output <dir>` — where to write SRTs + manifest (default: `<input_dir>/extracted_subs/`)
- `--ocr-language <lang>` — language hint for burned-in OCR (default: `pl`)
- `--verbose` / `-v`

### Translate additions

```
movie-translator <input> --external-subs <dir> [existing-flags...]
```

- `--external-subs <dir>` — directory containing SRTs + manifest from a prior `extract` run. Matching subtitles are added as additional tracks (never replace existing pipeline outputs).

## Extract Pipeline

For each video file discovered:

1. **Identify** — reuse `identify_media()` to resolve title, season, episode
2. **Probe tracks** — use `SubtitleExtractor.get_track_info()` to find embedded subtitle streams
3. **Extract text tracks** — for English (`eng`/`en`) and Polish (`pol`/`pl`) text-based subtitle streams, extract them directly via FFmpeg. Filter out signs/songs tracks using existing `_categorize_tracks()`.
4. **OCR burned-in** — if `--ocr-language` is set, run the upgraded burned-in extractor regardless of whether text tracks exist (the burned-in subs are a different source).
5. **Write output** — save SRT files with normalized naming, write/update manifest JSON.

### Output naming

Files are named based on resolved identity:
- `{title} - S{season:02d}E{episode:02d}.{lang}.srt` for episodes
- `{title}.{lang}.srt` for movies
- OCR results get `.ocr` suffix: `{title} - S01E01.pl.ocr.srt`

### Manifest format

```json
{
  "version": 1,
  "source_dir": "/path/to/source/videos",
  "entries": [
    {
      "source_file": "1.mp4",
      "identity": {
        "title": "Frieren: Beyond Journey's End",
        "parsed_title": "Frieren",
        "season": 1,
        "episode": 1,
        "media_type": "episode",
        "is_anime": true
      },
      "subtitles": [
        {
          "file": "Frieren - S01E01.pl.ocr.srt",
          "language": "pl",
          "method": "ocr_burned_in",
          "line_count": 267
        }
      ]
    }
  ]
}
```

## OCR Upgrade

The burned-in extractor gets three improvements:

### 1. Resolution scaling

Frames are scaled to 720p width (`min(1280, native_width)`) during extraction. This is done in the FFmpeg filter chain at near-zero cost and reduces disk usage by ~42% and change detection time by ~50% with no accuracy loss (benchmarked on Fririen S01E01).

### 2. Change detection optimization

Instead of OCR-ing every frame:
1. Extract frames at configurable FPS (default: 3)
2. Run pixel-diff change detection on all frames (numpy, fast)
3. OCR only frames where a subtitle transition was detected

At 3 FPS over a 24-min episode: ~4300 frames extracted, ~1200 transitions detected, OCR runs on ~28% of frames.

### 3. Language parameter

The OCR engine receives the user-specified language hint (`--ocr-language`) instead of hardcoded English. Apple Vision supports Polish recognition natively.

### Configurable constants

At the top of `burned_in_extractor.py`:
```python
OCR_EXTRACT_FPS = 3
OCR_SCALE_WIDTH = 1280  # 720p
OCR_CROP_RATIO = 0.25
OCR_CHANGE_THRESHOLD = 15.0
OCR_VARIANCE_THRESHOLD = 200.0
```

## Translate Integration (--external-subs)

When `--external-subs <dir>` is provided:

1. Load `manifest.json` from the directory at startup
2. In `CreateTracksStage`, after building the normal track list, look up matching entries by identity (title + season + episode)
3. For each matching external subtitle, add it as an additional `SubtitleFile` track with title like `"Polish (external)"` or `"English (external)"`
4. These tracks are always additive — they never replace fetched, AI-translated, or original tracks

The matching logic: compare `(parsed_title.lower(), season, episode)` from the manifest against the current video's identity. Fall back to filename stem matching if identity matching fails.

## Files Changed

### New files
- `movie_translator/extract.py` — extract pipeline and CLI

### Modified files
- `movie_translator/main.py` — subcommand routing, `--external-subs` flag
- `movie_translator/ocr/frame_extractor.py` — add `scale_width` parameter
- `movie_translator/ocr/burned_in_extractor.py` — change detection, scaling, language param, constants
- `movie_translator/stages/create_tracks.py` — external subtitle track integration
- `movie_translator/context.py` — add `external_subs_dir` to `PipelineConfig`

## Testing

- Run `movie-translator extract /Users/w/Downloads/Fririen/1.mp4 --ocr-language pl` and verify SRT output + manifest
- Verify existing `movie-translator <input>` translate flow is unchanged
- Verify `--external-subs` adds tracks to mux output
