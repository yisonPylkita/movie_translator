from movie_translator.subtitles import SubtitleExtractor


class TestTrackSelectionPriority:
    """Test that dialogue tracks are prioritized over signs/songs tracks."""

    def test_single_signs_track_rejected(self):
        """When only a signs track exists, it should be rejected (return None)."""
        extractor = SubtitleExtractor()

        track_info = {
            'tracks': [
                {
                    'id': 0,
                    'type': 'subtitles',
                    'codec': 'ass',
                    'properties': {'language': 'eng', 'track_name': 'English Signs/Songs'},
                }
            ]
        }

        result = extractor.find_english_track(track_info)
        assert result is None

    def test_prefers_dialogue_over_signs_when_both_present(self):
        """When both dialogue and signs tracks exist, dialogue should be selected."""
        extractor = SubtitleExtractor()

        track_info = {
            'tracks': [
                {
                    'id': 0,
                    'type': 'subtitles',
                    'codec': 'ass',
                    'properties': {'language': 'eng', 'track_name': 'English Signs/Songs'},
                },
                {
                    'id': 1,
                    'type': 'subtitles',
                    'codec': 'ass',
                    'properties': {'language': 'eng', 'track_name': 'English Full Dialogue'},
                },
            ]
        }

        result = extractor.find_english_track(track_info)
        assert result is not None
        assert result['id'] == 1
        assert 'Dialogue' in result['properties']['track_name']

    def test_prefers_dialogue_even_when_signs_listed_first(self):
        """Dialogue should be selected even if signs track is listed first."""
        extractor = SubtitleExtractor()

        track_info = {
            'tracks': [
                {
                    'id': 0,
                    'type': 'subtitles',
                    'codec': 'ass',
                    'properties': {'language': 'eng', 'track_name': 'Signs and Songs'},
                },
                {
                    'id': 1,
                    'type': 'subtitles',
                    'codec': 'ass',
                    'properties': {'language': 'eng', 'track_name': 'English'},
                },
            ]
        }

        result = extractor.find_english_track(track_info)
        assert result is not None
        assert result['id'] == 1
        assert result['properties']['track_name'] == 'English'

    def test_track_without_name_treated_as_dialogue(self):
        """Tracks without names should be treated as dialogue tracks."""
        extractor = SubtitleExtractor()

        track_info = {
            'tracks': [
                {
                    'id': 0,
                    'type': 'subtitles',
                    'codec': 'ass',
                    'properties': {'language': 'eng', 'track_name': 'Signs'},
                },
                {
                    'id': 1,
                    'type': 'subtitles',
                    'codec': 'ass',
                    'properties': {'language': 'eng', 'track_name': ''},
                },
            ]
        }

        result = extractor.find_english_track(track_info)
        assert result is not None
        assert result['id'] == 1

    def test_multiple_signs_tracks_all_rejected(self):
        """When only multiple signs/songs tracks exist, all should be rejected."""
        extractor = SubtitleExtractor()

        track_info = {
            'tracks': [
                {
                    'id': 0,
                    'type': 'subtitles',
                    'codec': 'ass',
                    'properties': {'language': 'eng', 'track_name': 'English Signs'},
                },
                {
                    'id': 1,
                    'type': 'subtitles',
                    'codec': 'ass',
                    'properties': {'language': 'eng', 'track_name': 'English Songs'},
                },
                {
                    'id': 2,
                    'type': 'subtitles',
                    'codec': 'ass',
                    'properties': {'language': 'eng', 'track_name': 'OP/ED'},
                },
            ]
        }

        result = extractor.find_english_track(track_info)
        assert result is None
