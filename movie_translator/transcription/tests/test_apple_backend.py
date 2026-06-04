"""Pure JSON-parse helper of the Apple bridge output."""

from movie_translator.transcription.apple_backend import _parse_segments
from movie_translator.types import DialogueLine


def test_parse_drops_empty_and_zero_duration():
    payload = (
        '{"segments":['
        '{"start_ms":0,"end_ms":0,"text":"timeless"},'      # no audioTimeRange runs
        '{"start_ms":100,"end_ms":50,"text":"inverted"},'   # never trust end<start
        '{"start_ms":10,"end_ms":20,"text":"  "},'
        '{"start_ms":10,"end_ms":900,"text":" keep me "}'
        ']}'
    )
    assert _parse_segments(payload) == [DialogueLine(10, 900, 'keep me')]
