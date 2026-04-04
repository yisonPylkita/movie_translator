from movie_translator.metrics.compare import compare_reports, match_videos


class TestMatchVideos:
    def test_match_by_hash(self):
        before = [{'hash': 'aaa', 'path': 'ep01.mkv', 'entries': []}]
        after = [{'hash': 'aaa', 'path': 'ep01.mkv', 'entries': []}]
        matched, excluded = match_videos(before, after)
        assert len(matched) == 1
        assert excluded == 0

    def test_match_by_identity(self):
        before = [
            {
                'hash': 'aaa',
                'path': 'ep01.mkv',
                'identity': {'media_type': 'episode', 'title': 'Test', 'season': 1, 'episode': 1},
                'entries': [{'name': 'identify', 'duration_ms': 10}],
            }
        ]
        after = [
            {
                'hash': 'bbb',
                'path': 'ep01_v2.mkv',
                'identity': {'media_type': 'episode', 'title': 'Test', 'season': 1, 'episode': 1},
                'entries': [{'name': 'identify', 'duration_ms': 8}],
            }
        ]
        matched, excluded = match_videos(before, after)
        assert len(matched) == 1

    def test_no_match(self):
        before = [{'hash': 'aaa', 'path': 'ep01.mkv', 'identity': {}, 'entries': []}]
        after = [{'hash': 'bbb', 'path': 'ep02.mkv', 'identity': {}, 'entries': []}]
        matched, excluded = match_videos(before, after)
        assert len(matched) == 0

    def test_exclude_different_profiles(self):
        before = [
            {
                'hash': 'aaa',
                'path': 'ep01.mkv',
                'entries': [{'name': 'identify', 'duration_ms': 10}],
            }
        ]
        after = [
            {
                'hash': 'aaa',
                'path': 'ep01.mkv',
                'entries': [
                    {'name': 'identify', 'duration_ms': 8},
                    {'name': 'extract_reference.extract_pgs_track', 'duration_ms': 5000},
                ],
            }
        ]
        matched, excluded = match_videos(before, after)
        assert len(matched) == 0
        assert excluded == 1


class TestCompareReports:
    def test_basic_comparison(self):
        before = {
            'version': 1,
            'git_commit': 'aaa',
            'dirty': False,
            'timestamp': '2026-03-28T12:00:00Z',
            'config': {'device': 'mps'},
            'videos': [
                {
                    'hash': 'x',
                    'path': 'ep01.mkv',
                    'total_duration_ms': 1000,
                    'entries': [
                        {'name': 'identify', 'duration_ms': 200},
                        {'name': 'translate', 'duration_ms': 800},
                    ],
                }
            ],
        }
        after = {
            'version': 1,
            'git_commit': 'bbb',
            'dirty': False,
            'timestamp': '2026-03-30T12:00:00Z',
            'config': {'device': 'mps'},
            'videos': [
                {
                    'hash': 'x',
                    'path': 'ep01.mkv',
                    'total_duration_ms': 800,
                    'entries': [
                        {'name': 'identify', 'duration_ms': 180},
                        {'name': 'translate', 'duration_ms': 620},
                    ],
                }
            ],
        }
        result = compare_reports(before, after)
        assert result['matched_videos'] == 1
        assert result['excluded_videos'] == 0
        spans = {s['name']: s for s in result['spans']}
        assert 'identify' in spans
        assert spans['identify']['before_ms'] == 200
        assert spans['identify']['after_ms'] == 180
        assert spans['identify']['delta_pct'] < 0

    def test_aggregates_multiple_videos(self):
        before = {
            'version': 1,
            'git_commit': 'a',
            'dirty': False,
            'config': {},
            'timestamp': '2026-03-28T12:00:00Z',
            'videos': [
                {
                    'hash': 'x',
                    'path': 'ep01.mkv',
                    'total_duration_ms': 100,
                    'entries': [{'name': 'identify', 'duration_ms': 10}],
                },
                {
                    'hash': 'y',
                    'path': 'ep02.mkv',
                    'total_duration_ms': 200,
                    'entries': [{'name': 'identify', 'duration_ms': 20}],
                },
            ],
        }
        after = {
            'version': 1,
            'git_commit': 'b',
            'dirty': False,
            'config': {},
            'timestamp': '2026-03-30T12:00:00Z',
            'videos': [
                {
                    'hash': 'x',
                    'path': 'ep01.mkv',
                    'total_duration_ms': 80,
                    'entries': [{'name': 'identify', 'duration_ms': 8}],
                },
                {
                    'hash': 'y',
                    'path': 'ep02.mkv',
                    'total_duration_ms': 160,
                    'entries': [{'name': 'identify', 'duration_ms': 16}],
                },
            ],
        }
        result = compare_reports(before, after)
        assert result['matched_videos'] == 2
        spans = {s['name']: s for s in result['spans']}
        assert spans['identify']['before_ms'] == 15
        assert spans['identify']['after_ms'] == 12
