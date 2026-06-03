"""Tests for the ilass alignment wiring (pure argv builder)."""

from __future__ import annotations

from pathlib import Path

from scripts.hardsub_poc.align import build_ilass_argv


def test_build_ilass_argv_matches_mt_fetch_order():
    argv = build_ilass_argv(
        Path('/bin/ilass'),
        Path('ref.srt'),
        Path('subs.srt'),
        Path('out.srt'),
    )
    # Same order as crates/mt-fetch/src/align_ilass.rs:
    # <binary> <reference> <incorrect> <output> --split-penalty 7.0 --disable-fps-guessing
    assert argv[0] == '/bin/ilass'
    assert argv[1] == 'ref.srt'  # reference (target timeline)
    assert argv[2] == 'subs.srt'  # subtitle to align
    assert argv[3] == 'out.srt'  # output
    assert argv[4:6] == ['--split-penalty', '7.0']
    assert '--disable-fps-guessing' in argv
