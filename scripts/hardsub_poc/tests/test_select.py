"""Tests for the resolved-player selection logic (pure, no I/O)."""

from __future__ import annotations

from scripts.hardsub_poc.__main__ import load_episodes, select_source


def _entry(host, sub, quality, embed='https://x/embed'):
    return {'host': host, 'sub': sub, 'quality': quality, 'embed_url': embed}


def test_prefers_cda_among_pl_subs():
    resolved = [
        _entry('sibnet', 'pl', '720p'),
        _entry('cda', 'pl', '1080p'),
        _entry('dood', 'pl', '720p'),
    ]
    chosen = select_source(resolved, sub='pl')
    assert chosen['host'] == 'cda'


def test_filters_by_sub_language():
    resolved = [
        _entry('cda', 'en', '1080p'),
        _entry('sibnet', 'pl', '720p'),
    ]
    chosen = select_source(resolved, sub='pl')
    assert chosen['host'] == 'sibnet'


def test_higher_resolution_wins_within_same_host():
    resolved = [
        _entry('cda', 'pl', '720p', embed='https://x/lo'),
        _entry('cda', 'pl', '1080p', embed='https://x/hi'),
    ]
    chosen = select_source(resolved, sub='pl')
    assert chosen['embed_url'] == 'https://x/hi'


def test_forced_host_overrides_preference():
    resolved = [
        _entry('cda', 'pl', '1080p'),
        _entry('sibnet', 'pl', '720p'),
    ]
    chosen = select_source(resolved, sub='pl', host='sibnet')
    assert chosen['host'] == 'sibnet'


def test_none_when_no_sub_match():
    resolved = [_entry('cda', 'en', '1080p')]
    assert select_source(resolved, sub='pl') is None


def test_sub_none_accepts_any_language():
    resolved = [_entry('voe', '', '1080p')]
    chosen = select_source(resolved, sub=None)
    assert chosen['host'] == 'voe'


def test_unknown_host_ranks_last_but_still_selectable():
    resolved = [_entry('weirdhost', 'pl', '1080p')]
    chosen = select_source(resolved, sub='pl')
    assert chosen['host'] == 'weirdhost'


def test_skips_entries_without_embed_url():
    resolved = [
        {'host': 'cda', 'sub': 'pl', 'quality': '1080p'},  # no embed_url
        _entry('sibnet', 'pl', '720p'),
    ]
    chosen = select_source(resolved, sub='pl')
    assert chosen['host'] == 'sibnet'


def test_load_episodes_multi_episode_shape():
    data = {
        'anime_slug': 'x',
        'episodes': [
            {'episode': 1, 'episode_url': 'u1', 'resolved': [_entry('cda', 'pl', '1080p')]},
            {'episode': 2, 'episode_url': 'u2', 'resolved': []},
        ],
    }
    eps = load_episodes(data)
    assert [e['episode'] for e in eps] == [1, 2]
    assert eps[0]['resolved'][0]['host'] == 'cda'


def test_load_episodes_flow_extract_flat_shape():
    data = {'episode_id': '205240', 'resolved': [_entry('cda', 'pl', '1080p')]}
    eps = load_episodes(data)
    assert len(eps) == 1
    assert eps[0]['episode'] == '205240'
    assert eps[0]['resolved'][0]['host'] == 'cda'
