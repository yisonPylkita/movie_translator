import subprocess

import pytest


@pytest.fixture
def create_test_frame(tmp_path):
    """Create test frames using FFmpeg drawtext."""
    from movie_translator.ffmpeg import get_ffmpeg

    ffmpeg = get_ffmpeg()

    def _create(filename, text=None, bg_color='black'):
        output = tmp_path / filename
        if text:
            vf = f"drawtext=text='{text}':fontsize=36:fontcolor=white:x=(w-tw)/2:y=(h-th)/2"
            cmd = [
                ffmpeg,
                '-y',
                '-f',
                'lavfi',
                '-i',
                f'color={bg_color}:s=640x120:d=1',
                '-vf',
                vf,
                '-frames:v',
                '1',
                str(output),
            ]
        else:
            cmd = [
                ffmpeg,
                '-y',
                '-f',
                'lavfi',
                '-i',
                f'color={bg_color}:s=640x120:d=1',
                '-frames:v',
                '1',
                str(output),
            ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            pytest.skip(f'Could not create test frame: {result.stderr}')
        return output

    return _create


class TestDetectTransitions:
    def test_no_transitions_for_identical_frames(self, create_test_frame):
        from movie_translator.ocr.change_detector import detect_transitions

        blank1 = create_test_frame('blank1.jpg')
        blank2 = create_test_frame('blank2.jpg')

        frames = [(blank1, 0), (blank2, 100)]
        transitions = detect_transitions(frames)

        assert len(transitions) == 0

    def test_detects_text_appearing(self, create_test_frame):
        from movie_translator.ocr.change_detector import detect_transitions

        blank = create_test_frame('blank.jpg')
        with_text = create_test_frame('text.jpg', text='Hello world')

        frames = [(blank, 0), (with_text, 100)]
        transitions = detect_transitions(frames)

        assert len(transitions) >= 1
        assert transitions[0].event_type == 'appeared'
        assert transitions[0].timestamp_ms == 100

    def test_detects_text_disappearing(self, create_test_frame):
        from movie_translator.ocr.change_detector import detect_transitions

        with_text = create_test_frame('text.jpg', text='Hello world')
        blank = create_test_frame('blank.jpg')

        frames = [(with_text, 0), (blank, 100)]
        transitions = detect_transitions(frames)

        assert len(transitions) >= 1
        assert transitions[-1].event_type == 'disappeared'

    def test_detects_text_change(self, create_test_frame):
        from movie_translator.ocr.change_detector import detect_transitions

        text1 = create_test_frame('text1.jpg', text='Hello world')
        text2 = create_test_frame('text2.jpg', text='Goodbye world')

        frames = [(text1, 0), (text2, 100)]
        transitions = detect_transitions(frames)

        assert len(transitions) >= 1
        assert transitions[0].event_type == 'appeared'

    def test_empty_frames_list(self):
        from movie_translator.ocr.change_detector import detect_transitions

        transitions = detect_transitions([])
        assert transitions == []

    def test_single_frame(self, create_test_frame):
        from movie_translator.ocr.change_detector import detect_transitions

        frame = create_test_frame('single.jpg', text='Hello')
        transitions = detect_transitions([(frame, 0)])

        assert transitions == []
