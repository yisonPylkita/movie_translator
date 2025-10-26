# movie_translator
Translate movies with subtitles to a different language


For now only extraction and apllication of already translated subtitles is supported. I'm using an AI website to translate those in batch - https://aisubtitletranslator.com/

It doesn't have a CLI tool so until then manual process it is


## Requirements

```bash
brew install mkvtoolnix
```


## Usage

```bash
# Extract subtitles from your movies
translate.bash extract

# MANUAL STEP - translate those extracted subtitles using https://aisubtitletranslator.com/

# Apply translated subtitles
translate.bash apply
```
