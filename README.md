# Movie Translator 🎬

Complete pipeline for extracting English dialogue from MKV files and translating to Polish using AI.

## Features 🚀

- **Smart Extraction**: Automatically finds English dialogue subtitle tracks (skips signs/songs)
- **AI Translation**: Uses `allegro/BiDi-eng-pol` model for high-quality English→Polish translation
- **Clean Output**: Creates MKV files with only 2 subtitle tracks (English dialogue + Polish translation)
- **MacBook Optimized**: MPS acceleration for Apple Silicon, optimized batch processing
- **Quality Code**: Linted with Ruff for clean, maintainable code
- **🎨 Fancy Terminal Output**: Beautiful Rich progress bars, spinners, and live updates

## Quick Start 🎯

### Setup (One-time)
```bash
cd /Users/arlen/h_dev/movie_translator
make setup  # Installs dependencies and runs linter
```

### Basic Usage
```bash
# Process single MKV file
uv run python3 translate.py ~/Downloads/test_movies/SPY\ x\ FAMILY\ -\ S01E01.mkv

# Process directory of MKV files
uv run python3 translate.py ~/Downloads/test_movies

# Custom output directory
uv run python3 translate.py ~/Downloads/test_movies --output ~/Downloads/translated_movies
```

### Advanced Options
```bash
# MacBook optimized (MPS + batch size)
uv run python3 translate.py ~/Downloads/test_movies --device mps --batch-size 16

# CPU processing (if MPS issues)
uv run python3 translate.py ~/Downloads/test_movies --device cpu --batch-size 8

# Larger batch size (faster but more memory)
uv run python3 translate.py ~/Downloads/test_movies --device mps --batch-size 32
```

## Development 🛠️

### Code Quality
```bash
# Lint code
make lint

# Format code
make format

# Run both lint and format
make check

# Clean temporary files
make clean
```

### Testing
```bash
# Quick test
make test

# Full pipeline test
make run-example
```

## 🎨 Terminal Output

The translator now features **beautiful Rich terminal output** with:

- **📊 Configuration Panels**: Clean tables showing your settings
- **⚡ Live Progress Bars**: Real-time progress for file processing and translation
- **🔄 Spinners**: Animated status indicators for model loading
- **📈 Batch Progress**: Step-by-step translation progress with time tracking
- **🎯 Summary Tables**: Clean results display with success/failure counts
- **🎨 Color Coding**: Beautiful colored output for different message types

### Example Output
```
┌─────────────────────────────────────────────────────────────┐
│               Movie Translator - Final Pipeline            │
├─────────────────────────────────────────────────────────────┤
│ Setting        │ Value                                      │
│ Input          │ /path/to/movies                           │
│ Output         │ /path/to/movies/translated                 │
│ Device         │ mps                                        │
│ Batch Size     │ 16                                         │
└─────────────────────────────────────────────────────────────┘

📥 Loading AI translation model...
🔤 Loading tokenizer...
🧠 Loading model...
📍 Moving model to mps...
🔧 Creating pipeline...
✅ Model loaded successfully

🔄 Translating 245 texts... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 00:32
127/245 lines • 3.9 lines/sec • 00:30 remaining • 00:32 elapsed

Processing 3 MKV files... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 02:15
├── Processing movie1.mkv... ✅
├── Processing movie2.mkv... ✅  
└── Processing movie3.mkv... ✅

┌─────────────────────────────────────────────────────────────┐
│                 Translation Complete                         │
├─────────────────────────────────────────────────────────────┤
│ ✅ Successful    │ 3                                          │
│ ❌ Failed        │ 0                                          │
│ 📁 Total         │ 3                                          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 🎉 All files processed successfully!                        │
│ 🎬 Clean MKVs with English dialogue + Polish translation   │
│ 📁 Output directory: /path/to/movies/translated           │
└─────────────────────────────────────────────────────────────┘
```

### 🚀 Enhanced Progress Features

The translation progress bar now shows **live real-time statistics**:

- **📊 Line Count**: `127/245 lines` - Current progress vs total
- **⚡ Processing Speed**: `3.9 lines/sec` - Live translation speed
- **⏰ Time Remaining**: `00:30 remaining` - Dynamic ETA calculation
- **⏱️ Elapsed Time**: `00:32 elapsed` - Time spent so far
- **🎯 Progress Bar**: Visual progress percentage

### 📈 Smart Time Estimation

- **Dynamic Calculation**: ETA updates based on current processing speed
- **Adaptive**: Adjusts to faster/slower batches automatically
- **Accurate**: Uses actual performance data, not estimates
- **Real-time**: Updates every batch for precision timing

## Pipeline Steps 📋

1. **📖 Extract**: Finds and extracts English dialogue subtitles (skips signs/songs)
2. **🔍 Filter**: Extracts only real dialogue lines from ASS files
3. **🤖 Translate**: AI translates dialogue to Polish using BiDi-eng-pol model
4. **🔨 Rebuild**: Creates clean English and Polish subtitle files
5. **🎬 Merge**: Builds clean MKV with only 2 subtitle tracks
6. **🔍 Verify**: Confirms perfect result
7. **🧹 Cleanup**: Removes temporary files

## Output 📁

- **Input**: MKV file(s) with multiple subtitle tracks
- **Output**: Clean MKV with exactly 2 tracks:
  - English Dialogue (original)
  - Polish (AI) (translated)
- **🚫 Removed**: All signs/songs tracks

## Requirements 📦

- **Python**: 3.10+ (3.14 has compatibility issues)
- **System Tools**: mkvmerge, mkvextract (from mkvtoolnix)
- **Python Packages**: pysubs2, torch, transformers, ruff

## Example Result 🎬

```bash
📁 Input:  /Users/arlen/Downloads/test_movies/SPY x FAMILY - S01E01.mkv
📁 Output: /Users/arlen/Downloads/test_movies/translated/SPY x FAMILY - S01E01_clean.mkv
🎬 Contains: English dialogue + Polish AI translation
🚫 Removed: All signs/songs tracks
```

## Translation Quality 🌐

The BiDi-eng-pol model provides high-quality translations:

- "How much longer to the embassy?" → "Ile jeszcze do ambasady?"
- "The brakes aren't working." → "Hamulce nie działają."
- "We must uncover their plot, no matter the cost." → "Musimy odkryć ich spisek, bez względu na koszty."

## Troubleshooting 🔧

### Common Issues
- **mkvmerge not found**: Install mkvtoolnix (`brew install mkvtoolnix` on macOS)
- **Memory errors**: Reduce batch size (`--batch-size 8`)
- **MPS errors**: Use CPU device (`--device cpu`)

### Dependencies
```bash
# Install all dependencies
make install

# Check if everything is working
uv run python3 translate.py --help
```

## Legacy Tools (Previous Version)

The project also includes legacy tools for SRT-based workflows:
- **`srt-translate`** - Translate a single SRT file
- **`srt-extract`** - Extract English subtitles from MKV files
- **`srt-translate-batch`** - Translate multiple SRT files
- **`srt-apply`** - Merge Polish subtitles back into MKV files
- **`srt-validate`** - Validate subtitle timing and structure

These are still available but the new `translate.py` pipeline is recommended for better quality and cleaner output.

## License 📄

MIT License - feel free to use and modify for your projects!
