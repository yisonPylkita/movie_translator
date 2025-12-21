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


❯ ./run.sh ~/Downloads/test_movies --verbose
🎬 Movie Translator

DEBUG    Attempting to acquire lock 4462506880 on /Users/arlen/h_dev/movie_translator/.venv/lib/python3.14/site-packages/static_ffmpeg/lock.file                                                          
DEBUG    Lock 4462506880 acquired on /Users/arlen/h_dev/movie_translator/.venv/lib/python3.14/site-packages/static_ffmpeg/lock.file                                                                       
DEBUG    Attempting to release lock 4462506880 on /Users/arlen/h_dev/movie_translator/.venv/lib/python3.14/site-packages/static_ffmpeg/lock.file                                                          
DEBUG    Lock 4462506880 released on /Users/arlen/h_dev/movie_translator/.venv/lib/python3.14/site-packages/static_ffmpeg/lock.file                                                                       
🎬 Movie Translator - 1 file(s)
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
DEBUG    at line 1: section heading [Script Info]                                                                                                                                                         
DEBUG    at line 18: section heading [Aegisub Project Garbage]                                                                                                                                            
DEBUG    at line 29: section heading [V4+ Styles]                                                                                                                                                         
DEBUG    at line 37: section heading [Events]                                                                                                                                                             
DEBUG    at line 1075: section heading [Aegisub Extradata]                                                                                                                                                
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
⠴ Jujutsu Kaisen - 001 - Ryoumen Sukuna.mkv ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% 0:00:00`torch_dtype` is deprecated! Use `dtype` instead!
⠸ Jujutsu Kaisen - 001 - Ryoumen Sukuna.mkv ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% 0:00:02
DEBUG    Short input at index 0: "Yes." (4 chars) - translation quality may vary                                                                                                                          
DEBUG    Short input at index 5: "Huh?" (4 chars) - translation quality may vary                                                                                                                          
DEBUG    Short input at index 5: "Wha—" (4 chars) - translation quality may vary                                                                                                                          
DEBUG    Short input at index 9: "Huh?" (4 chars) - translation quality may vary                                                                                                                          
DEBUG    Short input at index 13: "Yay!" (4 chars) - translation quality may vary                                                                                                                         
DEBUG    Short input at index 10: "Hm?" (3 chars) - translation quality may vary                                                                                                                          
DEBUG    Short input at index 7: "Huh?" (4 chars) - translation quality may vary                                                                                                                          
DEBUG    Short input at index 4: "Run!" (4 chars) - translation quality may vary                                                                                                                          
DEBUG    Short input at index 6: "Nue!" (4 chars) - translation quality may vary                                                                                                                          
INFO     Preprocessing Statistics:                                                                                                                                                                        
           Total lines processed: 392                                                                                                                                                                     
           Single-word matches: 5                                                                                                                                                                         
           Multi-word matches: 0                                                                                                                                                                          
           Idiom replacements: 0                                                                                                                                                                          
           Direct translation rate: 1.3% (skipped model)                                                                                                                                                  
INFO     🧹 Cleaning up AI Translator...                                                                                                                                                                  
DEBUG    at line 1: section heading [Script Info]                                                                                                                                                         
DEBUG    at line 18: section heading [Aegisub Project Garbage]                                                                                                                                            
DEBUG    at line 29: section heading [V4+ Styles]                                                                                                                                                         
DEBUG    at line 37: section heading [Events]                                                                                                                                                             
DEBUG    at line 1075: section heading [Aegisub Extradata]                                                                                                                                                
INFO        - Found 16 embedded font(s), checking Polish character support...                                                                                                                             
DEBUG    Reading 'cmap' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'cmap' table                                                                                                                                                                         
DEBUG    Reading 'post' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'post' table                                                                                                                                                                         
DEBUG    Reading 'maxp' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'maxp' table                                                                                                                                                                         
DEBUG    Reading 'cmap' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'cmap' table                                                                                                                                                                         
DEBUG    Reading 'cmap' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'cmap' table                                                                                                                                                                         
DEBUG    Reading 'post' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'post' table                                                                                                                                                                         
DEBUG    Reading 'maxp' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'maxp' table                                                                                                                                                                         
DEBUG    Reading 'cmap' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'cmap' table                                                                                                                                                                         
DEBUG    Reading 'post' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'post' table                                                                                                                                                                         
DEBUG    Reading 'maxp' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'maxp' table                                                                                                                                                                         
DEBUG    Reading 'cmap' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'cmap' table                                                                                                                                                                         
DEBUG    Reading 'cmap' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'cmap' table                                                                                                                                                                         
DEBUG    Reading 'CFF ' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'CFF ' table                                                                                                                                                                         
DEBUG    Reading 'cmap' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'cmap' table                                                                                                                                                                         
DEBUG    Reading 'post' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'post' table                                                                                                                                                                         
DEBUG    Reading 'maxp' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'maxp' table                                                                                                                                                                         
DEBUG    Reading 'cmap' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'cmap' table                                                                                                                                                                         
DEBUG    Reading 'post' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'post' table                                                                                                                                                                         
DEBUG    Reading 'maxp' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'maxp' table                                                                                                                                                                         
DEBUG    Reading 'cmap' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'cmap' table                                                                                                                                                                         
DEBUG    Reading 'post' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'post' table                                                                                                                                                                         
DEBUG    Reading 'maxp' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'maxp' table                                                                                                                                                                         
DEBUG    Reading 'cmap' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'cmap' table                                                                                                                                                                         
DEBUG    Reading 'post' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'post' table                                                                                                                                                                         
DEBUG    Reading 'maxp' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'maxp' table                                                                                                                                                                         
DEBUG    Reading 'cmap' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'cmap' table                                                                                                                                                                         
DEBUG    Reading 'post' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'post' table                                                                                                                                                                         
DEBUG    Reading 'maxp' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'maxp' table                                                                                                                                                                         
DEBUG    Reading 'cmap' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'cmap' table                                                                                                                                                                         
DEBUG    Reading 'post' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'post' table                                                                                                                                                                         
DEBUG    Reading 'maxp' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'maxp' table                                                                                                                                                                         
DEBUG    Reading 'cmap' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'cmap' table                                                                                                                                                                         
DEBUG    Reading 'CFF ' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'CFF ' table                                                                                                                                                                         
DEBUG    Reading 'cmap' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'cmap' table                                                                                                                                                                         
DEBUG    Reading 'post' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'post' table                                                                                                                                                                         
DEBUG    Reading 'maxp' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'maxp' table                                                                                                                                                                         
DEBUG    Reading 'cmap' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'cmap' table                                                                                                                                                                         
DEBUG    Reading 'cmap' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'cmap' table                                                                                                                                                                         
DEBUG    Reading 'CFF ' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'CFF ' table                                                                                                                                                                         
DEBUG    Reading 'cmap' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'cmap' table                                                                                                                                                                         
DEBUG    Reading 'post' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'post' table                                                                                                                                                                         
DEBUG    Reading 'maxp' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'maxp' table                                                                                                                                                                         
DEBUG    Reading 'cmap' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'cmap' table                                                                                                                                                                         
DEBUG    Reading 'post' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'post' table                                                                                                                                                                         
DEBUG    Reading 'maxp' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'maxp' table                                                                                                                                                                         
DEBUG    Reading 'cmap' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'cmap' table                                                                                                                                                                         
DEBUG    Reading 'post' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'post' table                                                                                                                                                                         
DEBUG    Reading 'maxp' table from disk                                                                                                                                                                   
DEBUG    Decompiling 'maxp' table                                                                                                                                                                         
INFO        - 4/16 embedded font(s) support Polish characters                                                                                                                                             
INFO     Creating subtitle files...                                                                                                                                                                       
INFO     🔨 Creating clean English ASS: Jujutsu Kaisen - 001 - Ryoumen Sukuna_english_clean.ass                                                                                                           
DEBUG    at line 1: section heading [Script Info]                                                                                                                                                         
DEBUG    at line 18: section heading [Aegisub Project Garbage]                                                                                                                                            
DEBUG    at line 29: section heading [V4+ Styles]                                                                                                                                                         
DEBUG    at line 37: section heading [Events]                                                                                                                                                             
DEBUG    at line 1075: section heading [Aegisub Extradata]                                                                                                                                                
INFO        - Saved 392 events                                                                                                                                                                            
INFO        - Removed all non-dialogue events                                                                                                                                                             
DEBUG    at line 1: section heading [Script Info]                                                                                                                                                         
DEBUG    at line 18: section heading [Aegisub Project Garbage]                                                                                                                                            
DEBUG    at line 29: section heading [V4+ Styles]                                                                                                                                                         
DEBUG    at line 37: section heading [Events]                                                                                                                                                             
DEBUG    at line 1075: section heading [Aegisub Extradata]                                                                                                                                                
DEBUG    at line 1: section heading [Script Info]                                                                                                                                                         
DEBUG    at line 18: section heading [V4+ Styles]                                                                                                                                                         
DEBUG    at line 26: section heading [Events]                                                                                                                                                             
INFO        📊 Validation: Original file has 1033 non-empty events                                                                                                                                        
INFO        📊 Validation: 400 dialogue, 633 non-dialogue (signs/songs/effects)                                                                                                                           
INFO        📊 Original dialogue range: 2570ms - 1425530ms (1423.0s)                                                                                                                                      
INFO           First: "Morning...."                                                                                                                                                                       
INFO           Last:  "..."                                                                                                                                                                               
INFO        📊 Validation: Cleaned file has 392 dialogue events                                                                                                                                           
INFO        📊 Cleaned dialogue range: 2570ms - 1424440ms (1421.9s)                                                                                                                                       
INFO           First: "Morning...."                                                                                                                                                                       
INFO           Last:  "I will exorcise you as a curse!..."                                                                                                                                                
INFO        📊 Timing differences:                                                                                                                                                                        
INFO           Start: +0ms ("within 50ms tolerance")                                                                                                                                                      
INFO           End:   -1090ms ("EXCEEDS 50ms tolerance")                                                                                                                                                  
ERROR       ❌ End time mismatch exceeds tolerance                                                                                                                                                        
ERROR          This likely means non-dialogue content (credits/signs) exists after last dialogue                                                                                                          
ERROR          Found 1 non-dialogue events after last dialogue:                                                                                                                                           
ERROR             1. [Signs] "Episode 2                                                                                                                                                                   
         For Myself..."                                                                                                                                                                                   
ERROR    Failed: Jujutsu Kaisen - 001 - Ryoumen Sukuna.mkv - Cleaned subtitles end time mismatch: 1424440ms vs 1425530ms (diff: -1090ms, tolerance: 50ms)                                                 
  Jujutsu Kaisen - 001 - Ryoumen Sukuna.mkv ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:12
✗ 1 failed

movie_translator on  main [!] is 📦 v1.0.0 via 🐍 v3.14.2 took 15s 
❯ 
