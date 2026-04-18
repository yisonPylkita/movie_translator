from movie_translator.subtitle_fetch.encoding import normalize_encoding


class TestNormalizeEncoding:
    def test_utf8_file_unchanged(self, tmp_path):
        p = tmp_path / 'test.srt'
        text = '1\n00:00:01,000 --> 00:00:02,000\nPółka z książkami\n'
        p.write_text(text, encoding='utf-8')
        normalize_encoding(p)
        assert p.read_text(encoding='utf-8') == text

    def test_cp1250_converted_to_utf8(self, tmp_path):
        p = tmp_path / 'test.srt'
        text = '1\n00:00:01,000 --> 00:00:02,000\nPółka z książkami\n'
        p.write_bytes(text.encode('cp1250'))
        normalize_encoding(p)
        result = p.read_text(encoding='utf-8')
        assert 'Półka' in result
        assert 'książkami' in result

    def test_iso_8859_2_converted_to_utf8(self, tmp_path):
        p = tmp_path / 'test.srt'
        text = '1\n00:00:01,000 --> 00:00:02,000\nŹródło świata\n'
        p.write_bytes(text.encode('iso-8859-2'))
        normalize_encoding(p)
        result = p.read_text(encoding='utf-8')
        assert 'Źródło' in result

    def test_utf8_bom_left_intact(self, tmp_path):
        p = tmp_path / 'test.srt'
        text = 'Test line'
        p.write_bytes(b'\xef\xbb\xbf' + text.encode('utf-8'))
        normalize_encoding(p)
        raw = p.read_bytes()
        assert raw.startswith(b'\xef\xbb\xbf')

    def test_ascii_file_unchanged(self, tmp_path):
        p = tmp_path / 'test.srt'
        text = 'Hello world'
        p.write_text(text, encoding='ascii')
        normalize_encoding(p)
        assert p.read_text() == text
