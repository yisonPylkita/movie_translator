"""Shared SRT writer (consolidates 3 prior per-module copies)."""

from __future__ import annotations

from movie_translator.srt import format_timestamp, write_srt
from movie_translator.types import DialogueLine


def test_format_timestamp():
    assert format_timestamp(0) == '00:00:00,000'
    assert format_timestamp(3_661_001) == '01:01:01,001'
    assert format_timestamp(59_999) == '00:00:59,999'


def test_write_srt_block_format(tmp_path):
    out = tmp_path / 'x.srt'
    write_srt(
        [DialogueLine(0, 1000, 'Hello'), DialogueLine(1500, 2500, 'World')],
        out,
    )
    assert out.read_text(encoding='utf-8') == (
        '1\n00:00:00,000 --> 00:00:01,000\nHello\n\n2\n00:00:01,500 --> 00:00:02,500\nWorld\n'
    )
