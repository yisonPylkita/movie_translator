# Movie Translator 🎬

[![Tests](https://github.com/yisonPylkita/movie_translator/actions/workflows/tests.yml/badge.svg)](https://github.com/yisonPylkita/movie_translator/actions/workflows/tests.yml)

**AI subtitle translator** for English→Polish translation. Works on macOS and Linux.

## Features

- **💻 Cross-Platform** - Works on macOS and Linux
- **🍎 MPS Acceleration** - Optimized for Apple Silicon
- **🤖 AI Translation** - Runs locally using pre-verified translation models
- **🎯 Smart Filtering** - Extracts dialogue only (skips signs/songs)
- **📦 Zero System Dependencies** - FFmpeg bundled via Python, no Homebrew needed

## Requirements

- **macOS** or **Linux**
- **[uv](https://docs.astral.sh/uv/)** - Python package manager
- That's it! FFmpeg is bundled automatically.

## Quick Start

### Setup (One-time)

```bash
# Option 1: Use the setup script
./setup.sh

# Option 2: Manual setup with uv
uv sync
```

### Usage

```bash
# Translate video files in a directory
./run.sh ~/Downloads/movies

# Adjust batch size for memory/speed tradeoff
./run.sh ~/Downloads/movies --batch-size 8

# Use CPU instead of MPS
./run.sh ~/Downloads/movies --device cpu

# Show all options
./run.sh --help
```

### With OCR Support (Optional)

```bash
# Install OCR dependencies
uv sync --extra ocr

# Process image-based subtitles
./run.sh ~/Downloads/movies --enable-ocr
```

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run linter
uv run ruff check .

# Run type checker
uv run ty check

# Run formatter
uv run ruff format .

# Run tests
uv run pytest -v
```

## License

MIT License - see LICENSE file for details.



INFO     Processing: Jujutsu Kaisen - 001 - Ryoumen Sukuna.mkv                                                                                                                                            
INFO     Extracting subtitles...                                                                                                                                                                          
INFO     Found 2 English subtitle track(s):                                                                                                                                                               
INFO       Track 1: ID=3, Name="Signs & Songs", Codec=ass                                                                                                                                                 
INFO       Track 2: ID=4, Name="Full Subtitles", Codec=ass                                                                                                                                                
INFO     Categorized: 1 dialogue track(s), 1 signs/songs track(s)                                                                                                                                         
INFO     Selected track: ID=4, Name="Full Subtitles"                                                                                                                                                      
INFO     Found English track: ID 4                                                                                                                                                                        
INFO     Extracting subtitle track 4...                                                                                                                                                                   
INFO     Extraction successful: Jujutsu Kaisen - 001 - Ryoumen Sukuna_extracted.ass                                                                                                                       
INFO     Parsing dialogue...                                                                                                                                                                              
INFO     📖 Reading Jujutsu Kaisen - 001 - Ryoumen Sukuna_extracted.ass...                                                                                                                                
INFO        - Loaded 1034 total events                                                                                                                                                                    
INFO        - Deduplicated: 1034 → 624 entries (removed 410 duplicate effect layers)                                                                                                                      
INFO        - Extracted 392 dialogue lines                                                                                                                                                                
INFO        - Skipped 232 non-dialogue events                                                                                                                                                             
INFO     Translating 392 lines...                                                                                                                                                                         
INFO     Initializing translator on mps                                                                                                                                                                   
INFO     Translation enhancements enabled (idioms, short phrases, cleanup)                                                                                                                                
INFO     Loading model...                                                                                                                                                                                 
/Users/arlen/h_dev/movie_translator/.venv/lib/python3.14/site-packages/transformers/models/marian/tokenization_marian.py:175: UserWarning: Recommended: pip install sacremoses.
  warnings.warn("Recommended: pip install sacremoses.")
⠦ Jujutsu Kaisen - 001 - Ryoumen Sukuna.mkv ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% 0:00:00`torch_dtype` is deprecated! Use `dtype` instead!
⠼ Jujutsu Kaisen - 001 - Ryoumen Sukuna.mkv ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% 0:00:04
INFO     Preprocessing Statistics:                                                                                                                                                                        
           Total lines processed: 392                                                                                                                                                                     
           Single-word matches: 5                                                                                                                                                                         
           Multi-word matches: 0                                                                                                                                                                          
           Idiom replacements: 0                                                                                                                                                                          
           Direct translation rate: 1.3% (skipped model)                                                                                                                                                  
INFO     🧹 Cleaning up AI Translator...                                                                                                                                                                  
INFO        - Found 16 embedded font(s), checking Polish character support...                                                                                                                             
INFO        - 4/16 embedded font(s) support Polish characters                                                                                                                                             
INFO     Creating subtitle files...                                                                                                                                                                       
INFO     🔨 Creating clean English ASS: Jujutsu Kaisen - 001 - Ryoumen Sukuna_english_clean.ass                                                                                                           
INFO        - Saved 392 events                                                                                                                                                                            
INFO        - Removed all non-dialogue events                                                                                                                                                             
ERROR    Failed: Jujutsu Kaisen - 001 - Ryoumen Sukuna.mkv - Cleaned subtitles end time mismatch: 1424440ms vs 1425530ms (tolerance: 50ms) 