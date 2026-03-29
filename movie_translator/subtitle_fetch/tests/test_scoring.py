from movie_translator.subtitle_fetch.scoring import compute_release_score


class TestReleaseScoring:
    def test_exact_match_scores_high(self):
        score = compute_release_score(
            'Breaking.Bad.S01E03.720p.BluRay',
            'Breaking.Bad.S01E03.720p.BluRay',
        )
        assert score >= 0.9

    def test_partial_match_scores_medium(self):
        score = compute_release_score(
            'Breaking.Bad.S01E03.720p.BluRay',
            'Breaking.Bad.S01E03.1080p.WEB',
        )
        assert 0.3 < score < 0.9

    def test_no_match_scores_low(self):
        score = compute_release_score(
            'Breaking.Bad.S01E03.720p.BluRay',
            'Totally.Different.Movie.2024',
        )
        assert score < 0.3

    def test_empty_strings(self):
        assert compute_release_score('', '') == 0.0
        assert compute_release_score('test', '') == 0.0

    def test_case_insensitive(self):
        s1 = compute_release_score('Breaking.Bad', 'breaking.bad')
        s2 = compute_release_score('Breaking.Bad', 'Breaking.Bad')
        assert s1 == s2
