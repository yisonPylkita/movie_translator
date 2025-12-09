from unittest.mock import patch

from movie_translator.pipeline import TranslationPipeline


def test_full_pipeline_with_ass_subtitles(create_test_mkv, tmp_output_dir):
    mkv_file = create_test_mkv(
        subtitle_content="""[Script Info]
Title: Test
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Hello world
Dialogue: 0,0:00:04.00,0:00:06.00,Default,,0,0,0,,How are you
""",
        language='eng',
        track_name='English',
    )

    mock_translated = [
        (1000, 3000, 'Witaj swiecie'),
        (4000, 6000, 'Jak sie masz'),
    ]

    pipeline = TranslationPipeline(device='cpu', batch_size=1, model='allegro')

    with patch.object(pipeline, '_translate', return_value=mock_translated):
        result = pipeline.process_video_file(mkv_file, tmp_output_dir)

    assert result is True
    assert mkv_file.exists()


def test_full_pipeline_with_srt_subtitles(create_test_mkv, tmp_output_dir):
    mkv_file = create_test_mkv(
        subtitle_content="""1
00:00:01,000 --> 00:00:03,000
Hello world

2
00:00:04,000 --> 00:00:06,000
How are you
""",
        subtitle_format='srt',
        language='eng',
        track_name='English',
    )

    mock_translated = [
        (1000, 3000, 'Witaj swiecie'),
        (4000, 6000, 'Jak sie masz'),
    ]

    pipeline = TranslationPipeline(device='cpu', batch_size=1, model='allegro')

    with patch.object(pipeline, '_translate', return_value=mock_translated):
        result = pipeline.process_video_file(mkv_file, tmp_output_dir)

    assert result is True
    assert mkv_file.exists()


def test_full_pipeline_detects_no_english_subtitles(create_test_mkv, tmp_output_dir):
    mkv_file = create_test_mkv(language='jpn', track_name='Japanese')

    pipeline = TranslationPipeline(device='cpu', batch_size=1)

    with patch.object(pipeline, '_translate', return_value=[]):
        result = pipeline.process_video_file(mkv_file, tmp_output_dir)

    assert result is False
