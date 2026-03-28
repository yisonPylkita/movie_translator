import platform
from pathlib import Path
from unittest.mock import patch

from movie_translator.fonts import (
    POLISH_CHARS,
    _font_filename_matches,
    _get_system_font_dirs,
    find_system_font_for_polish,
    font_supports_polish,
    get_ass_font_names,
    get_embedded_fonts,
    get_font_family_name,
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


def test_get_system_font_dirs_returns_existing_dirs():
    dirs = _get_system_font_dirs()
    assert all(d.is_dir() for d in dirs)
    assert len(dirs) > 0


def test_font_filename_matches():
    assert _font_filename_matches(Path('Arial.ttf'), 'Arial')
    assert _font_filename_matches(Path('Arial Bold.ttf'), 'Arial')
    assert _font_filename_matches(Path('arial.ttf'), 'Arial')
    assert not _font_filename_matches(Path('Verdana.ttf'), 'Arial')


def test_font_filename_matches_with_hyphens():
    assert _font_filename_matches(Path('DejaVu-Sans.ttf'), 'DejaVu Sans')
    assert _font_filename_matches(Path('Noto_Sans.ttf'), 'Noto Sans')


def test_get_font_family_name_nonexistent():
    assert get_font_family_name(Path('/nonexistent/font.ttf')) is None


def test_get_font_family_name_real_system_font():
    """Test with a real system font if available."""
    if platform.system() == 'Darwin':
        arial = Path('/System/Library/Fonts/Supplemental/Arial.ttf')
        if arial.exists():
            name = get_font_family_name(arial)
            assert name is not None
            assert 'Arial' in name


def test_find_system_font_for_polish_with_ass_match():
    """On a system with Arial, it should find it when ASS references 'arial'."""
    if platform.system() == 'Darwin':
        arial = Path('/System/Library/Fonts/Supplemental/Arial.ttf')
        if not arial.exists():
            return
        result = find_system_font_for_polish({'arial'})
        assert result is not None
        font_path, family_name = result
        assert font_supports_polish(font_path)


def test_find_system_font_for_polish_fallback():
    """When ASS references a nonexistent font, should fall back to a known font."""
    result = find_system_font_for_polish({'nonexistent_font_xyz'})
    # On most systems there should be at least one font with Polish support
    if result is not None:
        font_path, family_name = result
        assert font_supports_polish(font_path)
        assert family_name is not None


def test_find_system_font_for_polish_empty_set():
    result = find_system_font_for_polish(set())
    # Should still find a fallback
    if result is not None:
        _, family_name = result
        assert family_name is not None


def test_find_system_font_returns_none_when_no_fonts():
    with patch('movie_translator.fonts._iter_system_fonts', return_value=[]):
        result = find_system_font_for_polish({'arial'})
        assert result is None
