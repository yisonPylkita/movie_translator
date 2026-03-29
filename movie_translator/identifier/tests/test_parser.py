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


class TestAnimeDetection:
    def test_fansub_bracket_detected_as_anime(self):
        result = parse_filename('[HorribleSubs] Attack on Titan - 25 [1080p].mkv')
        assert result['is_anime'] is True
        assert result['release_group'] == 'HorribleSubs'
        assert result['title'] == 'Attack on Titan'
        assert result['episode'] == 25

    def test_subsplease_detected_as_anime(self):
        result = parse_filename('[SubsPlease] Chainsaw Man - 07 (1080p) [ABC12345].mkv')
        assert result['is_anime'] is True
        assert result['release_group'] == 'SubsPlease'
        assert result['title'] == 'Chainsaw Man'
        assert result['episode'] == 7

    def test_standard_tv_not_anime(self):
        result = parse_filename('Breaking.Bad.S01E03.720p.BluRay.mkv')
        assert result['is_anime'] is False
        assert result['release_group'] is None

    def test_anime_with_year(self):
        result = parse_filename('[TaigaSubs] Toradora! (2008) - 01 [720p].mkv')
        assert result['is_anime'] is True
        assert result['year'] == 2008
        assert result['episode'] == 1

    def test_anime_title_extraction(self):
        """Aniparse should extract cleaner anime titles than guessit."""
        result = parse_filename('[Erai-raws] Jujutsu Kaisen - 01 [1080p].mkv')
        assert result['title'] == 'Jujutsu Kaisen'
        assert result['is_anime'] is True
