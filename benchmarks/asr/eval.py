"""Scoring for the ASR bake-off. Runs in the isolated `eval` venv.

Pure, dependency-light helpers (`normalize_text`, `timing_errors`,
`join_text`) carry the custom logic and are unit-tested. WER/CER/chrF are thin
wrappers over jiwer / sacrebleu.
"""

from __future__ import annotations

import re
from pathlib import Path

# ASS inline override blocks: {\i1}, {\an8}, {\pos(..)} etc.
_ASS_TAG = re.compile(r'\{[^}]*\}')
# Punctuation to drop before WER/CER (ASCII + common CJK marks).
_PUNCT = re.compile(r'[.,!?;:"\'`()\[\]…—–\-。、！？「」『』（）〜・]+')
_WS = re.compile(r'\s+')


def normalize_text(s: str) -> str:
    """Lowercase, strip ASS override tags + punctuation, collapse whitespace."""
    s = _ASS_TAG.sub(' ', s)
    s = s.replace('\\N', ' ').replace('\\n', ' ')
    s = _PUNCT.sub(' ', s)
    s = _WS.sub(' ', s)
    return s.strip().lower()


def join_text(segments: list[dict]) -> str:
    """Normalize each segment's text and join into one normalized string."""
    parts = [normalize_text(s['text']) for s in segments]
    return _WS.sub(' ', ' '.join(p for p in parts if p)).strip()


def _overlap(a: dict, b: dict) -> int:
    return max(0, min(a['end_ms'], b['end_ms']) - max(a['start_ms'], b['start_ms']))


def timing_errors(ref_segs: list[dict], hyp_segs: list[dict]) -> dict:
    """Match hyp segments to ref by max temporal overlap; mean abs boundary error.

    Returns {matched, mean_start_err_ms, mean_end_err_ms}. A hyp segment with no
    temporal overlap against any ref segment is unmatched and excluded.
    """
    start_errs: list[int] = []
    end_errs: list[int] = []
    for h in hyp_segs:
        best = None
        best_ov = 0
        for r in ref_segs:
            ov = _overlap(h, r)
            if ov > best_ov:
                best_ov, best = ov, r
        if best is not None:
            start_errs.append(abs(h['start_ms'] - best['start_ms']))
            end_errs.append(abs(h['end_ms'] - best['end_ms']))
    matched = len(start_errs)
    return {
        'matched': matched,
        'mean_start_err_ms': round(sum(start_errs) / matched) if matched else 0,
        'mean_end_err_ms': round(sum(end_errs) / matched) if matched else 0,
    }


def parse_ass_segments(path: Path) -> list[dict]:
    """Parse an ASS/SRT file into [{start_ms, end_ms, text}] via pysubs2."""
    import pysubs2

    subs = pysubs2.load(str(path))
    out = []
    for ev in subs:
        if ev.is_comment or not ev.plaintext.strip():
            continue
        out.append({'start_ms': int(ev.start), 'end_ms': int(ev.end), 'text': ev.plaintext})
    return out


def wer(ref_text: str, hyp_text: str) -> float:
    import jiwer

    return float(jiwer.wer(ref_text, hyp_text))


def cer(ref_text: str, hyp_text: str) -> float:
    import jiwer

    return float(jiwer.cer(ref_text, hyp_text))


def chrf(hyps: list[str], refs: list[str]) -> float:
    import sacrebleu

    return float(sacrebleu.corpus_chrf(hyps, [refs]).score)
