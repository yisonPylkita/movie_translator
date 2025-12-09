from movie_translator.utils import clear_memory, replace_polish_chars


def test_replace_polish_chars_converts_lowercase():
    text = 'ąćęłńóśźż'
    result = replace_polish_chars(text)
    assert result == 'acelnoszz'


def test_replace_polish_chars_converts_uppercase():
    text = 'ĄĆĘŁŃÓŚŹŻ'
    result = replace_polish_chars(text)
    assert result == 'ACELNOSZZ'


def test_replace_polish_chars_preserves_non_polish():
    text = 'Hello World 123!'
    result = replace_polish_chars(text)
    assert result == 'Hello World 123!'


def test_replace_polish_chars_mixed_text():
    text = 'Cześć, jak się masz?'
    result = replace_polish_chars(text)
    assert result == 'Czesc, jak sie masz?'


def test_replace_polish_chars_empty_string():
    assert replace_polish_chars('') == ''


def test_clear_memory_runs_without_error():
    clear_memory()
