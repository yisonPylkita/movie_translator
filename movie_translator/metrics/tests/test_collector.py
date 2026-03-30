import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

from movie_translator.metrics.collector import MetricsCollector, NullCollector
from movie_translator.metrics.events import SpanEvent


class TestSpanEvent:
    def test_create_with_defaults(self):
        event = SpanEvent(name='fetch.download', duration_ms=1200.5)
        assert event.name == 'fetch.download'
        assert event.duration_ms == 1200.5
        assert event.details == {}

    def test_create_with_details(self):
        event = SpanEvent(
            name='translate.batch',
            duration_ms=85000,
            details={'input_lines': 342, 'batches': 22},
        )
        assert event.details['input_lines'] == 342
        assert event.details['batches'] == 22


class TestMetricsCollector:
    def test_span_emits_event_to_listener(self):
        collector = MetricsCollector()
        events = []
        collector.add_listener(events.append)

        with collector.span('identify'):
            pass

        assert len(events) == 1
        assert events[0].name == 'identify'
        assert events[0].duration_ms >= 0

    def test_span_records_duration(self):
        collector = MetricsCollector()
        events = []
        collector.add_listener(events.append)

        with collector.span('slow_op'):
            time.sleep(0.05)

        assert events[0].duration_ms >= 40

    def test_nested_spans_build_dotted_names(self):
        collector = MetricsCollector()
        events = []
        collector.add_listener(events.append)

        with collector.span('fetch'):
            with collector.span('search_all'):
                with collector.span('animesub'):
                    pass

        names = [e.name for e in events]
        assert 'fetch.search_all.animesub' in names
        assert 'fetch.search_all' in names
        assert 'fetch' in names

    def test_nested_spans_emit_inner_first(self):
        collector = MetricsCollector()
        events = []
        collector.add_listener(events.append)

        with collector.span('fetch'):
            with collector.span('download'):
                pass

        assert events[0].name == 'fetch.download'
        assert events[1].name == 'fetch'

    def test_span_detail_attaches_metadata(self):
        collector = MetricsCollector()
        events = []
        collector.add_listener(events.append)

        with collector.span('fetch.download') as s:
            s.detail('downloaded', 6)
            s.detail('failed', 1)

        assert events[0].details == {'downloaded': 6, 'failed': 1}

    def test_span_emits_on_exception(self):
        collector = MetricsCollector()
        events = []
        collector.add_listener(events.append)

        try:
            with collector.span('failing_op'):
                raise ValueError('boom')
        except ValueError:
            pass

        assert len(events) == 1
        assert events[0].name == 'failing_op'
        assert events[0].duration_ms >= 0

    def test_prefix_restored_after_exception(self):
        collector = MetricsCollector()
        events = []
        collector.add_listener(events.append)

        with collector.span('parent'):
            try:
                with collector.span('child'):
                    raise ValueError('boom')
            except ValueError:
                pass
            with collector.span('sibling'):
                pass

        names = [e.name for e in events]
        assert 'parent.child' in names
        assert 'parent.sibling' in names

    def test_parallel_spans_in_threads(self):
        collector = MetricsCollector()
        events = []
        collector.add_listener(events.append)

        def work_a():
            with collector.span('task_a'):
                time.sleep(0.01)

        def work_b():
            with collector.span('task_b'):
                time.sleep(0.01)

        with collector.span('parent'):
            # copy_context() captures the parent span prefix so that
            # worker threads inherit the dotted-path nesting.
            # Each worker needs its own copy since a Context cannot be
            # entered concurrently by two threads.
            with ThreadPoolExecutor(max_workers=2) as pool:
                fa = pool.submit(copy_context().run, work_a)
                fb = pool.submit(copy_context().run, work_b)
                fa.result()
                fb.result()

        names = [e.name for e in events]
        assert 'parent.task_a' in names
        assert 'parent.task_b' in names
        assert 'parent' in names

    def test_multiple_listeners(self):
        collector = MetricsCollector()
        events_a = []
        events_b = []
        collector.add_listener(events_a.append)
        collector.add_listener(events_b.append)

        with collector.span('op'):
            pass

        assert len(events_a) == 1
        assert len(events_b) == 1


class TestNullCollector:
    def test_span_is_noop(self):
        collector = NullCollector()
        with collector.span('anything') as s:
            s.detail('key', 'value')

    def test_add_listener_is_noop(self):
        collector = NullCollector()
        collector.add_listener(lambda e: None)

    def test_emit_is_noop(self):
        collector = NullCollector()
        collector.emit(SpanEvent(name='x', duration_ms=0))
