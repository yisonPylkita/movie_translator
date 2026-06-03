"""Unit tests for the ASR eval helpers. Run in the `eval` venv:

benchmarks/asr/envs/eval/bin/python -m pytest benchmarks/asr/test_eval.py
"""

from __future__ import annotations

import eval as E


def test_normalize_strips_ass_tags_punct_and_case():
    assert E.normalize_text(r'{\i1}Hello,{\i0} World!') == 'hello world'


def test_normalize_collapses_newlines_and_whitespace():
    assert E.normalize_text('a\\Nb   c\n d') == 'a b c d'


def test_join_text_normalizes_and_joins():
    segs = [{'text': 'Hello,'}, {'text': '{\\b1}World{\\b0}!'}]
    assert E.join_text(segs) == 'hello world'


def test_timing_errors_matches_by_overlap():
    ref = [
        {'start_ms': 0, 'end_ms': 1000, 'text': 'a'},
        {'start_ms': 1000, 'end_ms': 2000, 'text': 'b'},
    ]
    hyp = [
        {'start_ms': 100, 'end_ms': 1100, 'text': 'a'},  # overlaps ref[0]
        {'start_ms': 2100, 'end_ms': 3000, 'text': 'c'},  # overlaps nothing
    ]
    r = E.timing_errors(ref, hyp)
    assert r['matched'] == 1
    assert r['mean_start_err_ms'] == 100
    assert r['mean_end_err_ms'] == 100


def test_timing_errors_no_overlap_is_zero_matched():
    ref = [{'start_ms': 0, 'end_ms': 100, 'text': 'a'}]
    hyp = [{'start_ms': 500, 'end_ms': 600, 'text': 'a'}]
    r = E.timing_errors(ref, hyp)
    assert r['matched'] == 0
