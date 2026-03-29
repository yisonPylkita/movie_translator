"""Tests for subtitle_fetch.style_classifier module."""

import pysubs2

from movie_translator.subtitle_fetch.style_classifier import classify_styles


def _make_ass(styles_and_events: dict[str, list[tuple[int, int, str]]]) -> pysubs2.SSAFile:
    """Build an SSAFile with the given styles and events.

    Args:
        styles_and_events: {style_name: [(start_ms, end_ms, text), ...]}
            If text contains '\\pos(' it will be treated as positioned.
    """
    subs = pysubs2.SSAFile()
    for style_name, events in styles_and_events.items():
        subs.styles[style_name] = pysubs2.SSAStyle()
        for start, end, text in events:
            event = pysubs2.SSAEvent(start=start, end=end, text=text, style=style_name)
            subs.events.append(event)
    return subs


class TestClassifyStyles:
    def test_normal_dialogue_classified_as_dialogue(self):
        subs = _make_ass(
            {
                'Dialogue': [
                    (1000, 3000, 'Hello, world!'),
                    (4000, 6000, 'How are you doing today?'),
                    (7000, 9000, 'This is a normal subtitle line.'),
                ],
            }
        )
        assert 'Dialogue' in classify_styles(subs)

    def test_positioned_signs_classified_as_non_dialogue(self):
        subs = _make_ass(
            {
                'Signs': [
                    (1000, 3000, '{\\pos(960,100)}Location Name'),
                    (5000, 7000, '{\\pos(960,100)}Another Sign'),
                    (10000, 12000, '{\\pos(100,500)}Shop Name'),
                ],
            }
        )
        assert 'Signs' not in classify_styles(subs)

    def test_karaoke_short_text_classified_as_non_dialogue(self):
        # Per-character karaoke: many events, very short text
        events = [(i * 200, i * 200 + 150, c) for i, c in enumerate('abcdefghijklmnop' * 5)]
        subs = _make_ass({'OP-Romaji': events})
        assert 'OP-Romaji' not in classify_styles(subs)

    def test_rapid_fire_events_classified_as_non_dialogue(self):
        # 600 events, 300ms each — karaoke pattern
        events = [(i * 300, i * 300 + 250, f'syllable {i}') for i in range(600)]
        subs = _make_ass({'EDRO': events})
        assert 'EDRO' not in classify_styles(subs)

    def test_positioned_dialogue_rescued(self):
        """Dialogue with \\an8 top positioning should NOT be filtered."""
        subs = _make_ass(
            {
                'Dialogue Top': [
                    (1000, 3500, '{\\pos(960,50)}This is a top-positioned dialogue line'),
                    (4000, 6500, '{\\pos(960,50)}Another line spoken by a character'),
                    (7000, 9500, '{\\pos(960,50)}Third dialogue line with positioning'),
                ],
            }
        )
        # avg_text > 20, avg_dur > 1500 → rescued as dialogue
        assert 'Dialogue Top' in classify_styles(subs)

    def test_mixed_styles_classified_correctly(self):
        subs = _make_ass(
            {
                'Default': [
                    (1000, 3000, 'Normal dialogue here'),
                    (4000, 6000, 'More dialogue text for testing'),
                    (7000, 9000, 'Third line of regular dialogue'),
                ],
                'Signs': [
                    (1000, 3000, '{\\pos(960,100)}Location'),
                    (5000, 7000, '{\\pos(960,100)}Shop'),
                ],
                'OP': [(i * 200, i * 200 + 150, chr(65 + i % 26)) for i in range(80)],
            }
        )
        result = classify_styles(subs)
        assert 'Default' in result
        assert 'Signs' not in result
        assert 'OP' not in result

    def test_empty_subs_returns_empty(self):
        subs = pysubs2.SSAFile()
        assert classify_styles(subs) == set()

    def test_srt_default_style_is_dialogue(self):
        """SRT files produce a single 'Default' style — always dialogue."""
        subs = _make_ass(
            {
                'Default': [
                    (1000, 3000, 'First line of subtitles'),
                    (4000, 6000, 'Second line of subtitles'),
                    (7000, 9000, 'Third line of subtitles'),
                ],
            }
        )
        assert 'Default' in classify_styles(subs)

    def test_positioned_short_text_is_non_dialogue(self):
        """Positioned events with short text = signs, not dialogue."""
        subs = _make_ass(
            {
                'TypeSetting': [
                    (1000, 5000, '{\\pos(100,200)}EP 01'),
                    (6000, 10000, '{\\pos(100,200)}Title'),
                    (11000, 15000, '{\\pos(100,200)}Day 1'),
                ],
            }
        )
        assert 'TypeSetting' not in classify_styles(subs)
