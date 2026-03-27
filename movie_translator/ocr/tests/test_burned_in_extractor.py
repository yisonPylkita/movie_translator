from movie_translator.ocr.burned_in_extractor import _build_dialogue_lines_from_ocr, _write_srt
from movie_translator.types import DialogueLine


class TestBuildDialogueLinesFromOcr:
    def test_simple_text_then_blank(self):
        frame_texts = [
            (1000, 'Hello world'),
            (2000, 'Hello world'),
            (3000, ''),
        ]

        lines = _build_dialogue_lines_from_ocr(frame_texts)

        assert len(lines) == 1
        assert lines[0] == DialogueLine(1000, 3000, 'Hello world')

    def test_consecutive_different_texts(self):
        frame_texts = [
            (1000, 'First line'),
            (2000, 'First line'),
            (3000, 'Second line'),
            (4000, 'Second line'),
            (5000, ''),
        ]

        lines = _build_dialogue_lines_from_ocr(frame_texts)

        assert len(lines) == 2
        assert lines[0] == DialogueLine(1000, 3000, 'First line')
        assert lines[1] == DialogueLine(3000, 5000, 'Second line')

    def test_filters_empty_text(self):
        frame_texts = [
            (1000, ''),
            (2000, ''),
            (3000, 'Real text'),
            (4000, 'Real text'),
            (5000, ''),
        ]

        lines = _build_dialogue_lines_from_ocr(frame_texts)

        assert len(lines) == 1
        assert lines[0].text == 'Real text'

    def test_filters_single_character_garbage(self):
        frame_texts = [
            (1000, 'x'),
            (2000, ''),
        ]

        lines = _build_dialogue_lines_from_ocr(frame_texts)

        assert len(lines) == 0

    def test_deduplicates_consecutive_identical_text(self):
        frame_texts = [
            (1000, 'Same text'),
            (2000, 'Same text'),
            (3000, 'Same text'),
            (4000, 'Same text'),
            (5000, ''),
        ]

        lines = _build_dialogue_lines_from_ocr(frame_texts)

        assert len(lines) == 1
        assert lines[0] == DialogueLine(1000, 5000, 'Same text')

    def test_empty_input(self):
        lines = _build_dialogue_lines_from_ocr([])

        assert lines == []

    def test_last_subtitle_gets_extra_second(self):
        frame_texts = [
            (1000, 'Last line'),
            (2000, 'Last line'),
        ]

        lines = _build_dialogue_lines_from_ocr(frame_texts)

        assert len(lines) == 1
        assert lines[0] == DialogueLine(1000, 3000, 'Last line')


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
