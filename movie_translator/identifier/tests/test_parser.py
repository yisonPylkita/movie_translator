from movie_translator.identifier.parser import parse_filename


class TestParseFilename:
    def test_anime_bracket_format(self):
        result = parse_filename(
            '[One Pace][101-102] Reverse Mountain 01 [1080p][En Sub][583096D8].mp4'
        )
        assert result['title'] is not None
        assert isinstance(result['title'], str)

    def test_standard_tv_episode(self):
        result = parse_filename('Breaking.Bad.S01E03.720p.BluRay.x264.mkv')
        assert result['title'] == 'Breaking Bad'
        assert result['season'] == 1
        assert result['episode'] == 3

    def test_movie_with_year(self):
        result = parse_filename('Spirited.Away.2001.1080p.BluRay.mkv')
        assert result['title'] == 'Spirited Away'
        assert result['year'] == 2001
        assert result['media_type'] == 'movie'

    def test_episode_detected_as_episode_type(self):
        result = parse_filename('Naruto.S02E15.720p.mkv')
        assert result['media_type'] == 'episode'

    def test_folder_provides_series_context(self):
        result = parse_filename('Episode 01 [1080p].mkv', folder_name='One Piece')
        # Folder name should provide the series title
        assert result['title'] is not None

    def test_returns_none_for_missing_fields(self):
        result = parse_filename('random_video.mp4')
        assert result['season'] is None
        assert result['year'] is None
