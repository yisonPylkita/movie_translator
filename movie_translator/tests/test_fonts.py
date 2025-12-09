from pathlib import Path

from movie_translator.fonts import (
    POLISH_CHARS,
    font_supports_polish,
    get_ass_font_names,
    get_embedded_fonts,
)


def test_polish_chars_constant():
    assert 'ą' in POLISH_CHARS
    assert 'ć' in POLISH_CHARS
    assert 'ę' in POLISH_CHARS
    assert 'ł' in POLISH_CHARS
    assert 'Ą' in POLISH_CHARS
    assert len(POLISH_CHARS) == 18


def test_get_ass_font_names(create_ass_file):
    ass_file = create_ass_file()
    font_names = get_ass_font_names(ass_file)

    assert 'arial' in font_names


def test_get_ass_font_names_empty_file(tmp_path):
    empty_ass = tmp_path / 'empty.ass'
    empty_ass.write_text('')

    font_names = get_ass_font_names(empty_ass)
    assert font_names == set()


def test_get_embedded_fonts_no_fonts(create_test_mkv):
    mkv_file = create_test_mkv()
    fonts = get_embedded_fonts(mkv_file)

    assert fonts == []


def test_font_supports_polish_nonexistent_file():
    result = font_supports_polish(Path('/nonexistent/font.ttf'))
    assert result is False
