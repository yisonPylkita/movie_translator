from movie_translator.metrics.events import SpanEvent
from movie_translator.metrics.listeners import ReportBuilder


class TestReportBuilder:
    def test_start_and_end_video(self):
        rb = ReportBuilder()
        rb.start_video(
            path='Konosuba/01.mkv',
            hash='abc123',
            file_size_bytes=500_000_000,
            duration_ms=1_440_000,
            identity={'title': 'Konosuba', 'media_type': 'episode', 'season': 1, 'episode': 1},
        )
        rb.on_event(SpanEvent(name='identify', duration_ms=200))
        rb.end_video()

        assert len(rb.videos) == 1
        video = rb.videos[0]
        assert video['path'] == 'Konosuba/01.mkv'
        assert video['hash'] == 'abc123'
        assert video['file_size_bytes'] == 500_000_000
        assert video['duration_ms'] == 1_440_000
        assert video['identity']['title'] == 'Konosuba'
        assert len(video['entries']) == 1
        assert video['entries'][0]['name'] == 'identify'
        assert video['entries'][0]['duration_ms'] == 200

    def test_entries_only_in_current_video(self):
        rb = ReportBuilder()
        rb.start_video(
            path='ep01.mkv', hash='aaa', file_size_bytes=100, duration_ms=1000, identity={}
        )
        rb.on_event(SpanEvent(name='identify', duration_ms=10))
        rb.end_video()

        rb.start_video(
            path='ep02.mkv', hash='bbb', file_size_bytes=200, duration_ms=2000, identity={}
        )
        rb.on_event(SpanEvent(name='fetch', duration_ms=20))
        rb.end_video()

        assert len(rb.videos[0]['entries']) == 1
        assert rb.videos[0]['entries'][0]['name'] == 'identify'
        assert len(rb.videos[1]['entries']) == 1
        assert rb.videos[1]['entries'][0]['name'] == 'fetch'

    def test_entry_without_details_omits_key(self):
        rb = ReportBuilder()
        rb.start_video(path='ep.mkv', hash='x', file_size_bytes=1, duration_ms=1, identity={})
        rb.on_event(SpanEvent(name='op', duration_ms=5))
        rb.end_video()

        entry = rb.videos[0]['entries'][0]
        assert 'details' not in entry

    def test_entry_with_details_includes_them(self):
        rb = ReportBuilder()
        rb.start_video(path='ep.mkv', hash='x', file_size_bytes=1, duration_ms=1, identity={})
        rb.on_event(SpanEvent(name='op', duration_ms=5, details={'count': 3}))
        rb.end_video()

        entry = rb.videos[0]['entries'][0]
        assert entry['details'] == {'count': 3}

    def test_total_duration_set_on_end_video(self):
        rb = ReportBuilder()
        rb.start_video(path='ep.mkv', hash='x', file_size_bytes=1, duration_ms=1, identity={})
        rb.end_video()

        assert 'total_duration_ms' in rb.videos[0]
        assert rb.videos[0]['total_duration_ms'] >= 0

    def test_events_outside_video_are_ignored(self):
        rb = ReportBuilder()
        rb.on_event(SpanEvent(name='stray', duration_ms=1))
        assert len(rb.videos) == 0

    def test_update_current_video(self):
        rb = ReportBuilder()
        rb.start_video(path='ep.mkv', hash='', file_size_bytes=1, duration_ms=1, identity={})
        rb.update_current_video(identity={'title': 'Updated'}, hash='newhash')
        rb.end_video()

        assert rb.videos[0]['identity'] == {'title': 'Updated'}
        assert rb.videos[0]['hash'] == 'newhash'
