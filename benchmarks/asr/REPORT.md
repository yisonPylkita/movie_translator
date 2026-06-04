# ASR transcription bake-off — findings & recommendation

**Date:** 2026-06-04 · **Branch:** `poc/asr-transcription` · **Status:** PoC complete

Feasibility study for adding audio→subtitle transcription to the translator,
for anime with English or Japanese voiceover. 7 engines, one dual-audio episode
(`Isekai Ojisan S01E01`: JP + EN-dub audio + EN/PL reference subs) plus a clean
English pair (`oppenheimer`, 58 s English narration + matching transcript).
Hardware: this Mac (Apple Silicon, MPS available, macOS 26.5).

## TL;DR

- **Transcription is feasible and good.** On clean English every engine scores
  WER ≤ 0.05. On Japanese the large-v3 Whisper backends converge to
  near-identical, accurate transcripts, and Apple SpeechAnalyzer matches them
  (it even got 昏睡状態 "coma" right where *every* Whisper engine wrote the wrong
  homophone 渾水状態).
- **Recommendation: Apple SpeechAnalyzer as the primary engine, with
  mlx-whisper large-v3 as the cross-platform / fallback engine.** Apple is the
  fastest (RTF 0.025–0.11), needs no model download, has ~0 app-side RAM, is
  on-device, handles EN + JA, and fits the project's existing Apple stack
  (Vision OCR, Apple Translation). Its one cost: coarse segmentation that needs
  a sentence-splitting post-pass for subtitle-sized lines.
- **Avoid for this use:** WhisperX on Japanese (alignment collapses — 6–9
  segments for a 180 s clip); faster/openai large-v3 are CPU-bound and slow
  (RTF 2–3) on this Mac.

## How to read the metrics (honesty first)

- **EN-dub vs EN-sub WER is NOT an accuracy measure.** The Judas English
  subtitle is a *translation of the Japanese*, not a dub transcript — the dub
  rewrites wording ("You're Takafumi" vs the sub's "You must be Takafumi"). So
  Isekai EN WER (~0.9–1.7) measures dub-vs-sub divergence, not ASR error. The
  honest EN accuracy number comes from **oppenheimer** (clean audio+transcript).
- **oppenheimer is easy** (58 s of clear studio narration), so absolute WER
  there is optimistic. Treat it as "can the engine do clean English" (yes, all)
  and use it for relative speed/timing, not as anime-difficulty accuracy.
- **No Japanese ground truth exists.** JA quality is judged by (a) cross-engine
  agreement, (b) a manual read against the English reference, which *is* the
  translation of that audio. Reported as such, never as measured WER.
- **Timing error is confounded by segmentation granularity.** An engine that
  emits few long segments (Apple, kotoba) shows larger boundary error against a
  fine-grained reference even when timing is fine — judge timing together with
  segment count.

## Speed (realtime factor, lower = faster; 180 s segment)

| Engine | small EN | small JA | large-v3 EN | large-v3 JA | accel |
| --- | --- | --- | --- | --- | --- |
| apple-speech | — | — | **0.025** | **0.114** | ANE (OS) |
| mlx-whisper | 0.058 | 0.130 | 0.835 | 0.279 | Metal |
| whisper.cpp | 0.161 | 0.078 | 0.791 | 0.578 | Metal |
| faster-whisper | 0.186 | 0.131 | 2.173 | 2.174 | CPU int8 |
| openai-whisper | 0.191 | 0.265 | 3.090 | 0.684 | MPS→CPU* |
| whisperx | 0.708 | 1.094 | 0.413 | 0.581 | CPU+align |
| kotoba-whisper | — | 0.586 | — | — | MPS |

\* openai-whisper large-v3 EN hit unimplemented MPS ops and fell back to CPU
(RTF 3.09). Not practical at large-v3 on this Mac.

## Clean English accuracy (oppenheimer, 58 s)

| Engine / model | WER | CER | chrF | timing err (ms) | RAM (MB) |
| --- | --- | --- | --- | --- | --- |
| apple-speech | 0.014 | 0.005 | 98.6 | 2176† | ~18‡ |
| faster small | 0.014 | 0.005 | 98.8 | 1836 | 1112 |
| faster large-v3 | 0.000 | 0.000 | 100 | 1680 | 3577 |
| mlx small | 0.054 | 0.036 | 96.9 | 683 | 849 |
| mlx large-v3 | 0.041 | 0.047 | 99.0 | 1197 | 3488 |
| openai large-v3 | 0.041 | 0.047 | 99.0 | 1227 | 6002 |
| whisper.cpp small | 0.000 | 0.000 | 100 | 1219 | 746 |
| whisper.cpp large-v3 | 0.041 | 0.039 | 97.2 | 1995 | 3547 |
| **whisperx large-v3** | 0.000 | 0.000 | 100 | **55** | 4185 |
| whisperx small | 0.000 | 0.000 | 100 | **55** | 2495 |

† Apple's coarse segments inflate boundary error. ‡ App-side RSS — the model
runs in a system daemon, so Apple's true footprint is tiny vs Whisper's GBs.

**Read:** clean-English accuracy is a non-differentiator — everyone is excellent.
The differentiators are timing (WhisperX wins decisively via wav2vec2 forced
alignment, 55 ms), speed (Apple, then mlx/whisper.cpp), and footprint (Apple).

## Japanese (the real target)

The four large-v3 Whisper backends (mlx / openai / faster / whisper.cpp)
produced **near-identical** transcripts — they converge on the same output, which
is the strongest available signal of correctness absent ground truth. Manual
read against the English reference confirms accuracy (sister's-kid Takafumi /
"you've grown up" / family fell apart / "prove I was in another world" / the
invented spells イキュラス・キュオラ, ワーグレントセルド).

- **Apple SpeechAnalyzer**: accurate — uniquely got 昏睡状態 ("coma") correct
  where all Whisper engines wrote 渾水状態. But coarse: 16 segments vs ~58, so it
  needs sentence-splitting on 。 for subtitle lines.
- **kotoba-whisper (JP-specialist)**: underwhelmed — its 15 s chunking merged
  segments and it mangled the isekai line. No win over the large-v3 generalists
  here.
- **whisper-small (any backend)**: visibly worse than large-v3 on JA
  (mlx-small was the biggest consensus outlier, CER 0.63 vs others).
- **WhisperX on JA: broken** — wav2vec2 alignment collapsed to 6–9 segments with
  10 s+ timing errors. Do not use WhisperX for Japanese.

Shared error across all Whisper engines: 渾水状態 for 昏睡状態 — an audio-ambiguous
homophone. A glossary/LM post-pass (the project already has proper-noun handling)
would catch these.

## Per-engine verdict

| Engine | EN | JA | Speed | Footprint | Integration | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| **apple-speech** | excellent | excellent (coarse) | fastest | tiny, on-device | Swift bridge (have one) + sentence split | **primary** |
| **mlx-whisper** large-v3 | excellent | excellent | fast (Metal) | ~3.5 GB | pip, HF model | **fallback / cross-engine** |
| whisper.cpp large-v3 | excellent | excellent | fast (Metal) | ~3.5 GB | ggml, C++ dep | strong alt |
| faster-whisper | excellent | good@large | slow@large (CPU) | 1–3.5 GB | easy pip | ok for EN-small |
| openai-whisper | excellent | good@large | slow@large (MPS flaky) | up to 6 GB | reference only | baseline |
| whisperx | excellent + **best timing** | **broken** | medium | ~4 GB | heavy deps | EN-timing only |
| kotoba-whisper | n/a | mediocre here | medium | ~3 GB | transformers | not worth it |

## Recommendation

1. **Primary: Apple SpeechAnalyzer.** On-device, fastest, near-zero footprint,
   EN + JA, no model files to ship or download, and it slots beside the existing
   Apple Translation + Vision OCR backends. Required work: (a) the Swift bridge
   (a working PoC version exists at `adapters/apple_speech.swift`), built from
   source like the Apple Translation bridge; (b) a sentence-splitter to turn its
   coarse utterances into subtitle-sized lines (split on 。!? and length/duration
   caps). macOS-26-only — gate behind `is_available()` like the other Apple
   backends.
2. **Fallback: mlx-whisper large-v3.** For non-macOS-26 / when Apple is
   unavailable. Metal-accelerated, accurate on both languages, one pip dep + an
   HF model. Use large-v3 (small is noticeably worse on JA). **Must run VAD or
   OP/ED trimming in front of it** — bare Whisper hallucinates "thanks for
   watching" on the ED (see full-episode results). WhisperX bundles this VAD for
   English but breaks on Japanese alignment, so for JA use mlx + a standalone VAD
   (e.g. silero) rather than WhisperX.
3. **Optional EN timing polish:** WhisperX-style wav2vec2 forced alignment gives
   dramatically better English timestamps (55 ms). Only worth it if subtitle
   timing quality is a priority, EN-only, and never for JA.

This mirrors the existing pattern: Apple-native first (macOS), open fallback
second.

## Full-episode (drift / 24-min RTF) & end-to-end EN→PL

Ran the full 24-min episode (1422 s) to test long-audio behavior — this is where
the recommendation was confirmed and where Whisper showed its one real weakness.

| Engine / model / lang | RTF @24 min | segments | tail behavior |
| --- | --- | --- | --- |
| apple-speech EN | **0.010** | 408 | clean — no hallucination |
| apple-speech JA | **0.006** | 122 (coarse) | clean — no hallucination |
| mlx-whisper large-v3 EN | 0.82 | 1021 (over-segmented) | trails to empty/"you" |
| mlx-whisper large-v3 JA | 0.40 | 449 | **hallucinates** "ご視聴ありがとうございました" looped past audio end (1432 s > 1422 s) |
| faster-whisper small EN | 0.12 | 357 | clean |
| faster-whisper small JA | 0.21 | 496 | clean |
| whisperx large-v3 EN | 0.47 | 333 | clean (VAD) — timing 619 ms |

**The key long-audio finding:** Whisper large-v3 **hallucinates on the ED's
music/silence**, emitting the training-artifact phrase "thanks for watching"
(ご視聴ありがとうございました) on a loop with a timestamp past the end of the audio.
This is a well-known Whisper failure mode. Two engines avoid it: **WhisperX**
(VAD gates out non-speech) and **Apple SpeechAnalyzer** (native endpointing —
clean tails on both languages). mlx large-v3 also over-segments English (1021
fragments). **Production must run VAD/OP-ED trimming in front of any Whisper
backend; Apple needs none.**

Apple at full length is extraordinary: a 24-min episode in **9–14 s** (RTF
0.006–0.01), no drift, on-device.

**End-to-end EN-dub → PL** (mlx large-v3 EN transcript → production Allegro
EN→PL translator → chrF vs the real `pl_allegro` track): **chrF 56.7**, fluent
Polish ("Mój wujek został potrącony przez ciężarówkę, gdy miał zaledwie 17 lat").
This is confounded by dub≠sub wording (the transcript is of the *dub*, the
reference PL came from the *sub*), so it's a floor, not a degradation measure —
but it confirms ASR-sourced subtitles translate into clean, usable Polish.

**Net effect on the recommendation:** Apple SpeechAnalyzer's native endpointing
(no hallucination, no VAD plumbing, RTF ~0.01) widens its lead. The mlx fallback
needs VAD/OP-ED trimming to be production-safe on full episodes.

## Reproduce

See `README.md`. `prep_audio.py` → `run.py --variant {seg,opp,full}` →
`score.py`. Engine venvs + audio are gitignored; results JSON + refs committed.
