from movie_translator.subtitles.writer import _replace_polish_chars


def test_replace_polish_chars_converts_lowercase():
    text = 'ąćęłńóśźż'
    result = _replace_polish_chars(text)
    assert result == 'acelnoszz'


def test_replace_polish_chars_converts_uppercase():
    text = 'ĄĆĘŁŃÓŚŹŻ'
    result = _replace_polish_chars(text)
    assert result == 'ACELNOSZZ'


def test_replace_polish_chars_preserves_non_polish():
    text = 'Hello World 123!'
    result = _replace_polish_chars(text)
    assert result == 'Hello World 123!'


def test_replace_polish_chars_mixed_text():
    text = 'Cześć, jak się masz?'
    result = _replace_polish_chars(text)
    assert result == 'Czesc, jak sie masz?'


def test_replace_polish_chars_empty_string():
    assert _replace_polish_chars('') == ''
