"""swift_bridge: staleness logic without invoking swiftc."""

from __future__ import annotations

import os

import pytest

from movie_translator.swift_bridge import ensure_compiled


def test_fresh_binary_skips_compile(tmp_path, monkeypatch):
    src = tmp_path / 'b.swift'
    binary = tmp_path / 'b'
    src.write_text('// src')
    binary.write_text('bin')
    # binary newer than source -> no compile, swiftc never looked up
    os.utime(src, (1000, 1000))
    os.utime(binary, (2000, 2000))
    monkeypatch.setattr('shutil.which', lambda _: pytest.fail('must not look up swiftc'))
    assert ensure_compiled(src, binary) == binary


def test_stale_binary_without_swiftc_raises(tmp_path, monkeypatch):
    src = tmp_path / 'b.swift'
    binary = tmp_path / 'b'
    src.write_text('// src')
    binary.write_text('bin')
    os.utime(binary, (1000, 1000))
    os.utime(src, (2000, 2000))  # source newer -> stale
    monkeypatch.setattr('shutil.which', lambda _: None)
    with pytest.raises(FileNotFoundError, match='swiftc'):
        ensure_compiled(src, binary)


def test_missing_source_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match='source not found'):
        ensure_compiled(tmp_path / 'nope.swift', tmp_path / 'b')
