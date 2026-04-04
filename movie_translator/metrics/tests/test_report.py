import json
from unittest.mock import patch

from movie_translator.metrics.report import build_report, load_report, save_report


class TestBuildReport:
    def test_includes_version(self):
        report = build_report(
            videos=[],
            config={'device': 'mps', 'batch_size': 16, 'model': 'allegro'},
        )
        assert report['version'] == 1

    def test_includes_git_commit(self):
        with patch('movie_translator.metrics.report._git_short_hash', return_value='abc1234'):
            with patch('movie_translator.metrics.report._git_is_dirty', return_value=False):
                report = build_report(videos=[], config={})
        assert report['git_commit'] == 'abc1234'
        assert report['dirty'] is False

    def test_includes_dirty_flag(self):
        with patch('movie_translator.metrics.report._git_short_hash', return_value='abc1234'):
            with patch('movie_translator.metrics.report._git_is_dirty', return_value=True):
                report = build_report(videos=[], config={})
        assert report['dirty'] is True

    def test_includes_config(self):
        config = {'device': 'mps', 'batch_size': 16, 'model': 'allegro'}
        report = build_report(videos=[], config=config)
        assert report['config'] == config

    def test_includes_videos(self):
        videos = [{'path': 'ep01.mkv', 'entries': []}]
        report = build_report(videos=videos, config={})
        assert report['videos'] == videos

    def test_includes_timestamp(self):
        report = build_report(videos=[], config={})
        assert 'timestamp' in report


class TestSaveAndLoad:
    def test_roundtrip(self, tmp_path):
        report = {
            'version': 1,
            'git_commit': 'abc1234',
            'dirty': False,
            'timestamp': '2026-03-30T14:00:00Z',
            'config': {'device': 'mps'},
            'videos': [
                {
                    'path': 'ep01.mkv',
                    'hash': 'aaa',
                    'file_size_bytes': 100,
                    'duration_ms': 1000,
                    'identity': {'title': 'Test'},
                    'total_duration_ms': 500,
                    'entries': [{'name': 'identify', 'duration_ms': 10}],
                }
            ],
        }
        path = tmp_path / 'report.json'
        save_report(report, path)
        loaded = load_report(path)
        assert loaded == report

    def test_saved_file_is_readable_json(self, tmp_path):
        report = {'version': 1, 'videos': [], 'config': {}}
        path = tmp_path / 'report.json'
        save_report(report, path)
        raw = path.read_text()
        parsed = json.loads(raw)
        assert parsed['version'] == 1
