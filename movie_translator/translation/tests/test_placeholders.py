from movie_translator.translation.enhancements import extract_placeholders, restore_placeholders


class TestExtractPlaceholders:
    def test_phone_number_extracted(self):
        text = 'Call 555-123-4567 now.'
        result, mapping = extract_placeholders(text)
        assert '555-123-4567' not in result
        assert any('555-123-4567' in v for v in mapping.values())

    def test_url_extracted(self):
        text = 'Visit https://example.com for details.'
        result, mapping = extract_placeholders(text)
        assert 'https://example.com' not in result
        assert any('https://example.com' in v for v in mapping.values())

    def test_time_extracted(self):
        text = 'It happened at 12:30 today.'
        result, mapping = extract_placeholders(text)
        assert '12:30' not in result
        assert any('12:30' in v for v in mapping.values())

    def test_title_case_name_extracted(self):
        text = 'Talk to John Smith about it.'
        result, mapping = extract_placeholders(text)
        assert 'John Smith' not in result
        assert any('John Smith' in v for v in mapping.values())

    def test_no_placeholders_returns_unchanged(self):
        text = 'just a normal sentence.'
        result, mapping = extract_placeholders(text)
        assert result == text
        assert mapping == {}

    def test_empty_string(self):
        result, mapping = extract_placeholders('')
        assert result == ''
        assert mapping == {}

    def test_multiple_placeholders(self):
        text = 'John Smith called at 12:30.'
        result, mapping = extract_placeholders(text)
        assert len(mapping) == 2

    def test_placeholder_keys_are_unique(self):
        text = 'John Smith and Mary Jane met at 12:30 and 14:00.'
        _, mapping = extract_placeholders(text)
        keys = list(mapping.keys())
        assert len(keys) == len(set(keys))

    def test_stats_updated(self):
        from movie_translator.translation.enhancements import PreprocessingStats

        stats = PreprocessingStats()
        extract_placeholders('Call John Smith at 12:30.', stats)
        assert stats.placeholder_hits == 2


class TestRestorePlaceholders:
    def test_round_trip(self):
        text = 'Call 555-123-4567 or ask John Smith.'
        protected, mapping = extract_placeholders(text)
        restored = restore_placeholders(protected, mapping)
        assert restored == text

    def test_round_trip_url(self):
        text = 'Visit https://example.com/path?q=1 for info.'
        protected, mapping = extract_placeholders(text)
        restored = restore_placeholders(protected, mapping)
        assert restored == text

    def test_empty_mapping(self):
        result = restore_placeholders('hello world', {})
        assert result == 'hello world'

    def test_restore_with_surrounding_translation(self):
        # Simulate: model translated around the placeholder
        mapping = {'__NAME0__': 'John Smith'}
        translated = 'Porozmawiaj z __NAME0__ o tym.'
        restored = restore_placeholders(translated, mapping)
        assert restored == 'Porozmawiaj z John Smith o tym.'

    def test_placeholder_not_in_text_is_harmless(self):
        mapping = {'__NAME0__': 'John Smith'}
        result = restore_placeholders('no placeholder here', mapping)
        assert result == 'no placeholder here'
