import subprocess

import pytest

from movie_translator.ffmpeg import get_ffmpeg
from movie_translator.types import DialogueLine


@pytest.fixture
def tmp_output_dir(tmp_path):
    output_dir = tmp_path / 'output'
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def sample_srt_content():
    return """1
00:00:01,000 --> 00:00:03,000
Hello, how are you?

2
00:00:04,000 --> 00:00:06,000
I am fine, thank you.

3
00:00:10,000 --> 00:00:12,000
What a beautiful day!
"""


@pytest.fixture
def sample_ass_content():
    return """[Script Info]
Title: Test Subtitles
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1
Style: Signs,Arial,36,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,8,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Hello, how are you?
Dialogue: 0,0:00:04.00,0:00:06.00,Default,,0,0,0,,I am fine, thank you.
Dialogue: 0,0:00:07.00,0:00:09.00,Signs,,0,0,0,,{\\pos(960,100)}EPISODE 1
Dialogue: 0,0:00:10.00,0:00:12.00,Default,,0,0,0,,What a beautiful day!
"""


@pytest.fixture
def create_ass_file(tmp_path, sample_ass_content):
    def _create(filename='test.ass', content=None):
        ass_file = tmp_path / filename
        ass_file.write_text(content or sample_ass_content)
        return ass_file

    return _create


@pytest.fixture
def create_srt_file(tmp_path, sample_srt_content):
    def _create(filename='test.srt', content=None):
        srt_file = tmp_path / filename
        srt_file.write_text(content or sample_srt_content)
        return srt_file

    return _create


@pytest.fixture
def create_test_mkv(tmp_path):
    def _create(
        filename='test.mkv',
        duration_seconds=1,
        language='eng',
        track_name='English',
        subtitle_content=None,
        subtitle_format='ass',
    ):
        if subtitle_content is None:
            if subtitle_format == 'srt':
                subtitle_content = """1
00:00:01,000 --> 00:00:03,000
Hello world
"""
            else:
                subtitle_content = """[Script Info]
Title: Test
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Hello world
"""

        sub_ext = 'srt' if subtitle_format == 'srt' else 'ass'
        sub_file = tmp_path / f'temp_subs.{sub_ext}'
        sub_file.write_text(subtitle_content)

        mkv_file = tmp_path / filename
        ffmpeg = get_ffmpeg()

        cmd = [
            ffmpeg,
            '-y',
            '-f',
            'lavfi',
            '-i',
            f'color=black:s=320x240:d={duration_seconds}',
            '-f',
            'lavfi',
            '-i',
            f'anullsrc=r=44100:cl=mono:d={duration_seconds}',
            '-i',
            str(sub_file),
            '-c:v',
            'libx264',
            '-preset',
            'ultrafast',
            '-c:a',
            'aac',
            '-c:s',
            'srt' if subtitle_format == 'srt' else 'ass',
            '-metadata:s:s:0',
            f'language={language}',
            '-metadata:s:s:0',
            f'title={track_name}',
            str(mkv_file),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f'Failed to create test MKV: {result.stderr}')

        sub_file.unlink()
        return mkv_file

    return _create


@pytest.fixture
def sample_dialogue_lines():
    return [
        DialogueLine(1000, 3000, 'Hello, how are you?'),
        DialogueLine(4000, 6000, 'I am fine, thank you.'),
        DialogueLine(10000, 12000, 'What a beautiful day!'),
    ]


@pytest.fixture
def sample_translated_lines():
    return [
        DialogueLine(1000, 3000, 'Cześć, jak się masz?'),
        DialogueLine(4000, 6000, 'Dobrze, dziękuję.'),
        DialogueLine(10000, 12000, 'Jaki piękny dzień!'),
    ]
