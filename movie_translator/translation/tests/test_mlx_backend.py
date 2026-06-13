"""Tests for the MLX translation backend.

These tests require Apple Silicon (MLX) and the model weights to be present.
They are marked with @pytest.mark.slow and skipped on non-Mac/non-ARM systems.
"""

import platform
import sys

import pytest

from movie_translator.translation.mlx_backend import (
    BidiMLXModel,
    SinusoidalPositionEmbedding,
    _create_causal_mask,
    create_padding_mask,
    create_sinusoidal_embeddings,
    is_available,
)

_IS_APPLE_SILICON = sys.platform == 'darwin' and platform.machine() == 'arm64'

pytestmark = [
    pytest.mark.skipif(not _IS_APPLE_SILICON, reason='MLX requires Apple Silicon (macOS ARM)'),
    pytest.mark.slow,
]


def test_is_available() -> None:
    """MLX availability check should return bool."""
    result = is_available()
    assert isinstance(result, bool)


def test_create_sinusoidal_embeddings() -> None:
    """Sinusoidal embeddings have correct shape and halved structure."""
    emb = create_sinusoidal_embeddings(1024, 1024)
    assert emb.shape == (1024, 1024)

    # First half should be sin values (pos 0 = all zeros)
    sentinel = 512
    first_row_sin = emb[0, :sentinel]
    first_row_cos = emb[0, sentinel:]

    # At position 0: sin(0) = 0, cos(0) = 1
    assert abs(first_row_sin.sum()) < 1e-5, 'Sin half at pos 0 should be near-zero'
    # Cos at position 0 should be near 1
    assert abs(first_row_cos.mean() - 1.0) < 0.01, 'Cos half at pos 0 should be near 1.0'


def test_sinusoidal_position_embedding() -> None:
    """SinusoidalPositionEmbedding module returns correct shape."""
    spe = SinusoidalPositionEmbedding(1024, 1024)
    result = spe((2, 10))
    assert result.shape == (1, 10, 1024)

    result = spe((1, 50))
    assert result.shape == (1, 50, 1024)


def test_causal_mask() -> None:
    """Causal mask has correct shape and upper-triangular -inf."""
    mask = _create_causal_mask(4)
    assert mask.shape == (1, 1, 4, 4)

    m = mask[0, 0]  # (4, 4)

    # Diagonal and below should be 0
    for i in range(4):
        assert m[i, i] == 0.0, f'Diagonal at {i} should be 0'

    # Above diagonal should be -inf
    for i in range(4):
        for j in range(i + 1, 4):
            assert m[i, j] == float('-inf'), f'Upper triangle ({i},{j}) should be -inf'


def test_padding_mask() -> None:
    """create_padding_mask converts 1D mask to 4D additive mask."""
    import mlx.core as mx

    mask_1d = mx.array([[1.0, 1.0, 0.0, 0.0]])
    mask_4d = create_padding_mask(mask_1d)

    assert mask_4d.shape == (1, 1, 1, 4)

    m = mask_4d[0, 0, 0]
    assert m[0] == 0.0
    assert m[1] == 0.0
    assert m[2] == float('-inf')
    assert m[3] == float('-inf')


class TestBidiMLXModel:
    """Integration tests for the full MLX model (requires weights)."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        if not is_available():
            pytest.skip('MLX model weights not available')

    def test_create_model(self) -> None:
        """Model creates and loads weights successfully."""
        model = BidiMLXModel()
        model.load_mlx_weights()

        # Check core components exist
        assert model.embed_tokens is not None
        assert model.encoder is not None
        assert model.decoder is not None
        assert model.lm_head is not None

        # Weight shapes
        assert model.embed_tokens.weight.shape == (32000, 1024)
        assert model.lm_head.weight.shape == (32000, 1024)
        assert model.lm_head.bias.shape == (32000,)

    def test_forward_pass(self) -> None:
        """Encoder and decoder forward pass produce correct shapes."""
        import mlx.core as mx

        model = BidiMLXModel()
        model.load_mlx_weights()

        input_ids = mx.array([[5, 215, 28, 57, 624, 23, 2]], dtype=mx.int32)

        # Encode
        memory = model.encode(input_ids)
        assert memory.shape == (1, 7, 1024)

        # Decode first step
        decoder_ids = mx.full((1, 1), 1, dtype=mx.int32)
        causal_mask = _create_causal_mask(1)
        logits = model.decode(decoder_ids, memory, self_mask=causal_mask)
        assert logits.shape == (1, 1, 32000)

    def test_generate(self) -> None:
        """Greedy generation produces token IDs."""
        import mlx.core as mx

        model = BidiMLXModel()
        model.load_mlx_weights()

        input_ids = mx.array([[5, 6752, 50]], dtype=mx.int32)  # ">>pol<< Hello!"
        output_ids = model.generate(input_ids, max_new_tokens=20)
        assert output_ids.shape[0] == 1  # batch dim preserved
        assert output_ids.shape[1] > 1  # at least start token + one generated

    def test_translate_single(self) -> None:
        """Translate a single text."""
        model = BidiMLXModel()
        model.load_mlx_weights()

        results = model.translate(['Hello!'], max_new_tokens=20)
        assert len(results) == 1
        assert isinstance(results[0], str)
        assert len(results[0]) > 0

    def test_translate_batch(self) -> None:
        """Translate multiple texts in batch."""
        model = BidiMLXModel()
        model.load_mlx_weights()

        texts = ['Hello!', 'Good morning!', 'Thank you.']
        results = model.translate(texts, max_new_tokens=20, batch_size=2)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, str)
            assert len(r) > 0

    def test_translate_empty(self) -> None:
        """Empty input returns empty list."""
        model = BidiMLXModel()
        model.load_mlx_weights()

        results = model.translate([], max_new_tokens=20)
        assert results == []

    def test_tokenize_source(self) -> None:
        """Tokenize source adds language prefix and produces int32 IDs."""
        import mlx.core as mx

        model = BidiMLXModel()
        text = 'Hello!'
        ids, mask = model.tokenize_source([text])

        assert ids.dtype == mx.int32
        assert ids.shape[0] == 1
        # First token should be '>>pol<<' = 5
        assert ids[0, 0].item() == 5

    def test_decode_target(self) -> None:
        """Decode target handles EOS stripping."""
        import mlx.core as mx

        model = BidiMLXModel()
        model.load_mlx_weights()

        # tokens: [start=1, 'Witaj', '!', 'Witaj', '!', EOS=2, pad=1]
        token_ids = mx.array([[1, 12837, 50, 12837, 50, 2, 1, 1]], dtype=mx.int32)
        texts = model.decode_target(token_ids)
        assert len(texts) == 1
        assert isinstance(texts[0], str)
