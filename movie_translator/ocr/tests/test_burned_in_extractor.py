from pathlib import Path

from movie_translator.ocr.burned_in_extractor import _build_dialogue_lines, _write_srt
from movie_translator.ocr.change_detector import SubtitleTransition
from movie_translator.types import DialogueLine


class TestBuildDialogueLines:
    def test_simple_appeared_disappeared_pair(self):
        transitions = [
            SubtitleTransition(1000, Path('f1.jpg'), 'appeared'),
            SubtitleTransition(3000, Path('f2.jpg'), 'disappeared'),
        ]
        ocr_results = {Path('f1.jpg'): 'Hello world'}

        lines = _build_dialogue_lines(transitions, ocr_results)

        assert len(lines) == 1
        assert lines[0] == DialogueLine(1000, 3000, 'Hello world')

    def test_consecutive_appeared_events(self):
        transitions = [
            SubtitleTransition(1000, Path('f1.jpg'), 'appeared'),
            SubtitleTransition(3000, Path('f2.jpg'), 'appeared'),
            SubtitleTransition(5000, Path('f3.jpg'), 'disappeared'),
        ]
        ocr_results = {
            Path('f1.jpg'): 'First line',
            Path('f2.jpg'): 'Second line',
        }

        lines = _build_dialogue_lines(transitions, ocr_results)

        assert len(lines) == 2
        assert lines[0] == DialogueLine(1000, 3000, 'First line')
        assert lines[1] == DialogueLine(3000, 5000, 'Second line')

    def test_filters_empty_ocr_results(self):
        transitions = [
            SubtitleTransition(1000, Path('f1.jpg'), 'appeared'),
            SubtitleTransition(3000, Path('f2.jpg'), 'appeared'),
            SubtitleTransition(5000, Path('f3.jpg'), 'disappeared'),
        ]
        ocr_results = {
            Path('f1.jpg'): '',
            Path('f2.jpg'): 'Real text',
        }

        lines = _build_dialogue_lines(transitions, ocr_results)

        assert len(lines) == 1
        assert lines[0].text == 'Real text'

    def test_filters_single_character_garbage(self):
        transitions = [
            SubtitleTransition(1000, Path('f1.jpg'), 'appeared'),
            SubtitleTransition(3000, Path('f2.jpg'), 'disappeared'),
        ]
        ocr_results = {Path('f1.jpg'): 'x'}

        lines = _build_dialogue_lines(transitions, ocr_results)

        assert len(lines) == 0

    def test_deduplicates_consecutive_identical_text(self):
        transitions = [
            SubtitleTransition(1000, Path('f1.jpg'), 'appeared'),
            SubtitleTransition(2000, Path('f2.jpg'), 'appeared'),
            SubtitleTransition(5000, Path('f3.jpg'), 'disappeared'),
        ]
        ocr_results = {
            Path('f1.jpg'): 'Same text',
            Path('f2.jpg'): 'Same text',
        }

        lines = _build_dialogue_lines(transitions, ocr_results)

        assert len(lines) == 1
        assert lines[0] == DialogueLine(1000, 5000, 'Same text')

    def test_empty_transitions(self):
        lines = _build_dialogue_lines([], {})

        assert lines == []


class TestWriteSrt:
    def test_writes_valid_srt(self, tmp_path):
        lines = [
            DialogueLine(1000, 3000, 'Hello world'),
            DialogueLine(4000, 6500, 'Second line'),
        ]
        output = tmp_path / 'output.srt'

        _write_srt(lines, output)

        content = output.read_text()
        assert '1\n' in content
        assert '00:00:01,000 --> 00:00:03,000' in content
        assert 'Hello world' in content
        assert '2\n' in content
        assert '00:00:04,000 --> 00:00:06,500' in content
        assert 'Second line' in content
