from movie_translator.subtitles import (
    SubtitleExtractor,
    SubtitleParser,
    SubtitleValidator,
    SubtitleWriter,
)


class TestSubtitleParser:
    def test_extract_dialogue_lines_from_ass(self, create_ass_file):
        ass_file = create_ass_file()
        parser = SubtitleParser()

        lines = parser.extract_dialogue_lines(ass_file)

        assert len(lines) == 3
        assert lines[0][2] == 'Hello, how are you?'
        assert lines[1][2] == 'I am fine, thank you.'
        assert lines[2][2] == 'What a beautiful day!'

    def test_extract_dialogue_lines_from_srt(self, create_srt_file):
        srt_file = create_srt_file()
        parser = SubtitleParser()

        lines = parser.extract_dialogue_lines(srt_file)

        assert len(lines) == 3
        assert lines[0][2] == 'Hello, how are you?'
        assert lines[1][2] == 'I am fine, thank you.'
        assert lines[2][2] == 'What a beautiful day!'

    def test_filters_signs_songs_style(self, create_ass_file):
        ass_file = create_ass_file()
        parser = SubtitleParser()

        lines = parser.extract_dialogue_lines(ass_file)
        texts = [line[2] for line in lines]

        assert 'EPISODE 1' not in texts

    def test_returns_timing_tuples_from_ass(self, create_ass_file):
        ass_file = create_ass_file()
        parser = SubtitleParser()

        lines = parser.extract_dialogue_lines(ass_file)

        for start, end, text in lines:
            assert isinstance(start, int)
            assert isinstance(end, int)
            assert isinstance(text, str)
            assert end > start

    def test_returns_timing_tuples_from_srt(self, create_srt_file):
        srt_file = create_srt_file()
        parser = SubtitleParser()

        lines = parser.extract_dialogue_lines(srt_file)

        for start, end, text in lines:
            assert isinstance(start, int)
            assert isinstance(end, int)
            assert isinstance(text, str)
            assert end > start


class TestSubtitleWriter:
    def test_create_english_ass(self, create_ass_file, tmp_path, sample_dialogue_lines):
        original_ass = create_ass_file()
        output_path = tmp_path / 'english_clean.ass'
        writer = SubtitleWriter()

        writer.create_english_ass(original_ass, sample_dialogue_lines, output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert 'Hello, how are you?' in content

    def test_create_polish_ass_with_replacement(
        self, create_ass_file, tmp_path, sample_translated_lines
    ):
        original_ass = create_ass_file()
        output_path = tmp_path / 'polish.ass'
        writer = SubtitleWriter()

        writer.create_polish_ass(
            original_ass, sample_translated_lines, output_path, replace_chars=True
        )

        assert output_path.exists()
        content = output_path.read_text()
        assert 'Czesc' in content
        assert 'ę' not in content

    def test_create_polish_ass_without_replacement(
        self, create_ass_file, tmp_path, sample_translated_lines
    ):
        original_ass = create_ass_file()
        output_path = tmp_path / 'polish.ass'
        writer = SubtitleWriter()

        writer.create_polish_ass(
            original_ass, sample_translated_lines, output_path, replace_chars=False
        )

        assert output_path.exists()
        content = output_path.read_text()
        assert 'Cześć' in content


class TestSubtitleExtractor:
    def test_get_track_info(self, create_test_mkv):
        mkv_file = create_test_mkv(language='eng')
        extractor = SubtitleExtractor()

        track_info = extractor.get_track_info(mkv_file)

        assert 'tracks' in track_info
        assert len(track_info['tracks']) == 1

    def test_find_english_track(self, create_test_mkv):
        mkv_file = create_test_mkv(language='eng', track_name='English')
        extractor = SubtitleExtractor()

        track_info = extractor.get_track_info(mkv_file)
        eng_track = extractor.find_english_track(track_info)

        assert eng_track is not None
        assert eng_track['properties']['language'] == 'eng'

    def test_find_english_track_returns_none_for_non_english(self, create_test_mkv):
        mkv_file = create_test_mkv(language='jpn', track_name='Japanese')
        extractor = SubtitleExtractor()

        track_info = extractor.get_track_info(mkv_file)
        eng_track = extractor.find_english_track(track_info)

        assert eng_track is None

    def test_extract_subtitle_ass(self, create_test_mkv, tmp_path):
        mkv_file = create_test_mkv(language='eng', subtitle_format='ass')
        extractor = SubtitleExtractor()
        output_path = tmp_path / 'extracted.ass'

        result = extractor.extract_subtitle(mkv_file, 0, output_path, subtitle_index=0)

        assert result is True
        assert output_path.exists()

    def test_extract_subtitle_srt(self, create_test_mkv, tmp_path):
        mkv_file = create_test_mkv(language='eng', subtitle_format='srt')
        extractor = SubtitleExtractor()
        output_path = tmp_path / 'extracted.srt'

        result = extractor.extract_subtitle(mkv_file, 0, output_path, subtitle_index=0)

        assert result is True
        assert output_path.exists()


class TestSubtitleValidator:
    def test_validate_matching_subtitles(self, create_ass_file, tmp_path):
        original = create_ass_file('original.ass')

        cleaned_content = """[Script Info]
Title: Test
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Hello, how are you?
Dialogue: 0,0:00:04.00,0:00:06.00,Default,,0,0,0,,I am fine, thank you.
Dialogue: 0,0:00:10.00,0:00:12.00,Default,,0,0,0,,What a beautiful day!
"""
        cleaned = tmp_path / 'cleaned.ass'
        cleaned.write_text(cleaned_content)

        validator = SubtitleValidator()
        result = validator.validate_cleaned_subtitles(original, cleaned)

        assert result is True
