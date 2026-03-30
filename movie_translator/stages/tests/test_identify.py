from unittest.mock import patch

from movie_translator.context import PipelineConfig, PipelineContext
from movie_translator.stages.identify import IdentifyStage


class TestIdentifyStage:
    def _make_ctx(self, tmp_path):
        video = tmp_path / 'ep01.mkv'
        video.touch()
        return PipelineContext(
            video_path=video,
            work_dir=tmp_path / 'work',
            config=PipelineConfig(),
        )

    def test_sets_identity(self, tmp_path):
        ctx = self._make_ctx(tmp_path)

        class FakeIdentity:
            title = 'Test'

        with patch('movie_translator.stages.identify.identify_media', return_value=FakeIdentity()):
            result = IdentifyStage().run(ctx)

        assert result.identity is not None
        assert result.identity.title == 'Test'  # ty: ignore[unresolved-attribute]

    def test_returns_same_context_object(self, tmp_path):
        ctx = self._make_ctx(tmp_path)

        with patch('movie_translator.stages.identify.identify_media', return_value=object()):
            result = IdentifyStage().run(ctx)

        assert result is ctx
