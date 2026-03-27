from movie_translator.ocr.burned_in_extractor import _build_dialogue_lines_from_ocr, _write_srt
from movie_translator.types import BoundingBox, DialogueLine


class TestMapBoxToFullFrame:
    def test_maps_crop_coordinates_to_full_frame(self):
        from movie_translator.ocr.burned_in_extractor import _map_box_to_full_frame

        # Box in middle of crop (crop_ratio=0.25, crop covers bottom 25%)
        crop_box = BoundingBox(x=0.1, y=0.3, width=0.8, height=0.2)
        result = _map_box_to_full_frame(crop_box, crop_ratio=0.25)

        assert result.x == 0.1  # x unchanged
        assert result.width == 0.8  # width unchanged
        # y should map from crop-space to full-frame: 0.75 + 0.3*0.25 = 0.825
        assert abs(result.y - 0.825) < 1e-9
        # height scales by crop_ratio: 0.2 * 0.25 = 0.05
        assert abs(result.height - 0.05) < 1e-9

    def test_top_of_crop_maps_to_crop_start(self):
        from movie_translator.ocr.burned_in_extractor import _map_box_to_full_frame

        crop_box = BoundingBox(x=0.0, y=0.0, width=1.0, height=0.1)
        result = _map_box_to_full_frame(crop_box, crop_ratio=0.25)

        # y=0 in crop → y=0.75 in full frame (top of bottom 25%)
        assert abs(result.y - 0.75) < 1e-9

    def test_bottom_of_crop_maps_to_frame_bottom(self):
        from movie_translator.ocr.burned_in_extractor import _map_box_to_full_frame

        crop_box = BoundingBox(x=0.0, y=0.9, width=1.0, height=0.1)
        result = _map_box_to_full_frame(crop_box, crop_ratio=0.25)

        # y=0.9 in crop → 0.75 + 0.9*0.25 = 0.975
        assert abs(result.y - 0.975) < 1e-9
        # height: 0.1 * 0.25 = 0.025, so bottom edge at 0.975+0.025 = 1.0
        assert abs(result.y + result.height - 1.0) < 1e-9


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
