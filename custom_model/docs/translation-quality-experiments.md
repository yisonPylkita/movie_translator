# Translation Quality Experiments

**Date:** 2026-04-04
**Test material:** Berserk (1997) S01E11 — 182 dialogue lines
**Backends tested:** Allegro BiDi (allegro/BiDi-eng-pol), Apple Translation (macOS 26+)

## Motivation

Both translation backends produce errors that make subtitles awkward or incomprehensible. The worst class of error is character name translation — "Guts" becomes "Wnętrzności" (intestines), "Casca" becomes "Kaska". Military commands like "March!" become "Marzec!" (the month). These are not edge cases; they happen on nearly every line containing a proper noun.

This document records systematic experiments to improve translation quality.

## Methodology

We extracted 182 English dialogue lines from Berserk S01E11 and ran each through 11 experimental configurations. We scored results against 27 diagnostic lines that cover known problem categories: proper nouns, idioms, military terms, gender agreement, archaic language, and complex grammar.

The automated scorer checks for:

- Untranslated English words (excluding proper nouns)
- Known mistranslations (Guts→Wnętrzności, March→Marzec, strike→strajk)
- Broken grammar (random mid-sentence capitalization)

Lower issue count = better. The scorer is conservative; it catches hard errors but not stylistic issues.

## Results

| Experiment | Backend | Time | Issues (of 27 lines) | Key changes |
|---|---|---|---|---|
| allegro_baseline | Allegro | 13.7s | 28 | Current defaults |
| allegro_beams4 | Allegro | 26.6s | 28 | num_beams=4 |
| allegro_beams4_lenpen | Allegro | 26.2s | 28 | num_beams=4, length_penalty=1.2 |
| allegro_name_protected | Allegro | 13.3s | 29 | Proper noun placeholders |
| allegro_names_beams4 | Allegro | 26.9s | 27 | Names + beams |
| **allegro_full** | **Allegro** | **27.9s** | **26** | Names + beams + glossary |
| apple_baseline | Apple | 4.5s | 30 | Current defaults |
| apple_name_protected | Apple | 4.4s | 27 | Proper noun placeholders |
| **apple_full** | **Apple** | **4.3s** | **25** | **Names + glossary (best)** |
| apple_context_window | Apple | 13.9s | 38 | Surrounding lines as context |
| apple_full_context | Apple | 13.8s | 30 | Names + glossary + context |

## What Works

### 1. Proper noun protection (biggest win)

Replace known character names with placeholder tokens (`__PN0__`) before translation, restore them after. This completely eliminates name translation errors:

| Line | Before | After |
|---|---|---|
| Guts... | Wnętrzności... | **Guts...** |
| Guts! | Wnętrzności! | **Guts!** |
| Casca! | Kaska! | **Casca!** |

**Implementation:** Extended `extract_placeholders()` to accept a `proper_nouns: set[str]` parameter. Names are detected automatically from subtitle content by `extract_proper_nouns_from_subtitles()`, which uses heuristics: words in direct address ("Guts!"), after honorifics ("Sir Griffith"), and capitalized words appearing 3+ times mid-sentence.

### 2. Glossary pre/post-processing (second biggest win)

A dictionary of context-dependent words and idioms that both models consistently mistranslate:

**Pre-translation** (exact match bypass):

- `march` → `naprzód` (military command, not the month)
- `charge` → `do ataku` (military, not battery/loading)
- `way to go` → `tak trzymaj`
- `get it together` → `weź się w garść`
- `son of a bitch` → `sukinsyn`

**Post-translation** (pattern fix on model output):

- `Marzec` → `Naprzód`
- `Dobra droga` → `Tak trzymaj`
- `Wnętrzności` → `Guts` (safety net)
- `strajk` → `cios` (labour strike → combat strike)
- `Weź to razem` → `Weź się w garść`

## What Doesn't Work

### 3. Beam search (Allegro only)

Changing from greedy decoding (`num_beams=1`) to beam search (`num_beams=4`) with length penalty:

- **Doubles translation time** (13s → 27s)
- **Minimal quality improvement** (28 → 27 issues with name protection)
- The model's vocabulary and training data are the bottleneck, not the search strategy
- A few lines get marginally better phrasing, but the same systemic errors persist

**Verdict:** Not worth the 2x speed cost for production use.

### 4. Context windows (Apple only)

Providing surrounding lines as context via `|||` separators:

- **Worst experiment** — issue count jumped from 30 to 38
- Apple Translation is a black-box sentence translator; it translates the entire block as one unit
- The separator sometimes survives but extraction is unreliable — lines bleed into each other
- Context actually confuses the model rather than helping it

**Verdict:** Do not use. Apple Translation cannot be steered with context.

## Remaining Unsolved Problems

Even with the best configuration (`apple_full`, 25 issues):

| Problem | Example | Root cause |
|---|---|---|
| `strike` → `strajk` | "blocking my strike" → "blokowanie mojego strajku" | Ambiguous word; model picks labour meaning over combat |
| `lodestone` untranslated | "necklace made from lodestone" | Rare English word not in model vocabulary |
| Gender agreement (Allegro) | "you are dressed" → "ubrany" (masc.) for Casca | Model has no character gender context |
| `class` → `klasa` | "mindful of your class" → "swojej klasie" | Should be "stanie" (social standing) |

These are addressable by expanding the glossary and post-processing rules — they're dictionary problems, not model architecture problems.

## Backend Comparison

### Apple Translation advantages

- **3x faster** (4.3s vs 13-28s)
- **Better grammar** — correct sentence boundaries, fewer broken sentences
- **Better gender agreement** — uses feminine "ubrana" for Casca (Allegro uses masculine)
- **More natural Polish** — reads less like word-by-word translation

### Allegro BiDi advantages

- **Runs offline** — no dependency on macOS version
- **Configurable** — beam search, temperature, max tokens are tunable
- **Open weights** — can be fine-tuned on domain-specific data

### Overall scores (1-10 scale, manual assessment)

| Criterion | Allegro baseline | Allegro full | Apple baseline | Apple full |
|---|---|---|---|---|
| Grammar | 3 | 4 | 6 | 6 |
| Natural Polish | 3 | 4 | 5 | 6 |
| Proper nouns | 1 | 8 | 1 | 8 |
| Idioms | 2 | 5 | 2 | 7 |
| Gender | 4 | 4 | 7 | 7 |
| **Overall** | **2.5** | **5** | **4.5** | **7** |

## Recommendation

**Use Apple Translation with the full enhancement pipeline** as the default backend when available. Fall back to Allegro with name protection for non-macOS systems.

The enhancement pipeline (proper nouns + glossary) provides the largest quality improvement per effort. Expanding the glossary is the straightforward path for further improvements.

## Files Changed

- `translation/enhancements.py` — added military commands to phrase map, idioms to multi-word phrases, post-translation fix patterns, proper noun support in `extract_placeholders()`
- `translation/proper_nouns.py` — new module for automatic character name detection from subtitle content
- `translation/translator.py` — `proper_nouns` attribute on `SubtitleTranslator`, threaded through `translate_dialogue_lines()`
- `translation/apple_backend.py` — `proper_nouns` attribute on `AppleTranslationBackend`
- `stages/translate.py` — calls `extract_proper_nouns_from_subtitles()` before translation
- `async_pipeline.py` — same for async path
- `gpu_queue.py` — `proper_nouns` field on `TranslateTask`

## Reproduction

The experiment script is at `scripts/translation_experiments.py`. To reproduce:

```bash
# Extract English lines from a translated video
ffmpeg -i video.mkv -map 0:s:0 /tmp/english.ass
# ... parse dialogue lines to /tmp/berserk_compare/english_lines.txt

# Run all experiments
python scripts/translation_experiments.py
```

Results are saved to `/tmp/berserk_compare/experiments/`.
