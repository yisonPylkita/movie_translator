"""MLX-accelerated translation backend for the Allegro BiDi-eng-pol model.

This module implements the same MarianMT-style encoder-decoder transformer
using Apple's MLX framework, which runs natively on Metal (Apple Silicon GPU)
without the PyTorch MPS bridge overhead.

Architecture: MarianMT (OPUS-MT lineage)
  - 6 encoder layers, 6 decoder layers
  - d_model=1024, 16 attention heads
  - FFN hidden dim=4096, ReLU activation
  - Post-LayerNorm (norm_first=False in MLX convention)
  - Sinusoidal position embeddings (static, not learned)
  - Shared encoder/decoder token embeddings with scale (sqrt(d_model))
  - SentencePiece tokenizer (32000 vocab, shared source/target)
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..logging import logger

# ---------------------------------------------------------------------------
# Constants — match the Allegro BiDi model config
# ---------------------------------------------------------------------------
D_MODEL = 1024
NUM_HEADS = 16
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
FFN_DIM = 4096  # encoder_ffn_dim / decoder_ffn_dim
DROPOUT = 0.1
VOCAB_SIZE = 32000
MAX_POSITION_EMBEDDINGS = 1024
PAD_TOKEN_ID = 1
EOS_TOKEN_ID = 2
DECODER_START_TOKEN_ID = 1  # pad_token_id
EMBED_SCALE = math.sqrt(D_MODEL)

MODEL_PATH = Path(__file__).parent.parent.parent / 'models' / 'allegro'
SAFETENSORS_PATH = MODEL_PATH / 'model.safetensors'
SOURCE_SPM_PATH = MODEL_PATH / 'source.spm'
TARGET_SPM_PATH = MODEL_PATH / 'target.spm'


# ---------------------------------------------------------------------------
# Sinusoidal position embeddings (Marian-compatible)
# ---------------------------------------------------------------------------


def create_sinusoidal_embeddings(num_positions: int, dim: int) -> mx.array:
    """Create sinusoidal position embeddings matching Marian's implementation.

    Marian pools cos features in the 2nd half of the vector rather than
    interleaving them.  The first ``dim // 2`` columns contain sin of
    even-indexed frequency terms; the remaining columns contain cos of
    odd-indexed frequency terms.
    """
    n_pos = num_positions
    sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1

    # (n_pos, dim) where each entry = pos / 10000^(2*(j//2)/dim)
    position_enc_np = np.array(
        [[pos / np.power(10000.0, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)],
        dtype=np.float32,
    )
    out = mx.zeros((n_pos, dim), dtype=mx.float32)
    out[:, 0:sentinel] = mx.sin(mx.array(position_enc_np[:, 0::2]))
    out[:, sentinel:] = mx.cos(mx.array(position_enc_np[:, 1::2]))
    return out


class SinusoidalPositionEmbedding(nn.Module):
    """Marian-compatible sinusoidal position embedding.

    Produces sinusoidal positional embeddings of any length up to
    MAX_POSITION_EMBEDDINGS. Pre-computed and constant (not trainable).
    """

    def __init__(self, num_positions: int, dim: int):
        super().__init__()
        weight = create_sinusoidal_embeddings(num_positions, dim)
        # Store as a Module parameter (frozen, no grad)
        self.weight = weight

    def __call__(self, shape: tuple[int, int]) -> mx.array:
        """Return position embeddings for (batch, seq_len)."""
        _bsz, seq_len = shape[:2]
        return self.weight[:seq_len][None, :, :]  # (1, seq_len, dim)


# ---------------------------------------------------------------------------
# BiDi model — MLX implementation of MarianMT
# ---------------------------------------------------------------------------


class BiDiEncoderLayer(nn.Module):
    """Single encoder layer matching Marian's post-LN design with biases."""

    def __init__(self, dims: int, num_heads: int, mlp_dims: int, dropout: float):
        super().__init__()
        self.self_attention = nn.MultiHeadAttention(dims, num_heads, bias=True)
        self.self_attention_layer_norm = nn.LayerNorm(dims)
        self.fc1 = nn.Linear(dims, mlp_dims)
        self.fc2 = nn.Linear(mlp_dims, dims)
        self.final_layer_norm = nn.LayerNorm(dims)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        # Self-attention (post-LN)
        residual = x
        x = self.self_attention(x, x, x, mask)
        x = self.dropout(x)
        x = residual + x
        x = self.self_attention_layer_norm(x)

        # FFN (post-LN)
        residual = x
        x = self.fc1(x)
        x = nn.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = residual + x
        x = self.final_layer_norm(x)
        return x


class BiDiEncoder(nn.Module):
    """Encoder stack: embedding + sinusoidal PE + N encoder layers + final LN."""

    def __init__(
        self,
        num_layers: int,
        dims: int,
        num_heads: int,
        mlp_dims: int,
        dropout: float,
        embed_tokens: nn.Embedding,
    ):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.embed_positions = SinusoidalPositionEmbedding(MAX_POSITION_EMBEDDINGS, dims)
        self.layers = [
            BiDiEncoderLayer(dims, num_heads, mlp_dims, dropout) for _ in range(num_layers)
        ]
        self.dropout = nn.Dropout(dropout)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        # Token embeddings (scaled)
        x = self.embed_tokens(input_ids) * EMBED_SCALE

        # Add position embeddings
        x = x + self.embed_positions(x.shape)

        x = self.dropout(x)

        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, attention_mask)

        return x


class BiDiDecoderLayer(nn.Module):
    """Single decoder layer matching Marian's post-LN design with biases.

    Supports KV-caching: if *cache* is provided, only the *new* token's
    hidden state (x[:, -1:, :]) is processed through self-attention, and
    cached K/V tensors are appended with the new token's projections.
    This avoids O(T²) recomputation during autoregressive generation.
    """

    def __init__(self, dims: int, num_heads: int, mlp_dims: int, dropout: float):
        super().__init__()
        self.self_attention = nn.MultiHeadAttention(dims, num_heads, bias=True)
        self.self_attention_layer_norm = nn.LayerNorm(dims)
        self.encoder_attention = nn.MultiHeadAttention(dims, num_heads, bias=True)
        self.encoder_attention_layer_norm = nn.LayerNorm(dims)
        self.fc1 = nn.Linear(dims, mlp_dims)
        self.fc2 = nn.Linear(mlp_dims, dims)
        self.final_layer_norm = nn.LayerNorm(dims)
        self.dropout = nn.Dropout(dropout)

    def __call__(
        self,
        x: mx.array,
        memory: mx.array,
        self_mask: mx.array | None = None,
        memory_mask: mx.array | None = None,
        cache: dict | None = None,
    ) -> mx.array:
        # ── Self-attention (post-LN) with fixed-buffer KV-cache ──
        residual = x

        if cache is not None:
            k = self.self_attention.key_proj(x)
            v = self.self_attention.value_proj(x)
            q = self.self_attention.query_proj(x)

            step = cache['step']
            # Slice-write into pre-allocated fixed buffers (no concatenate)
            cache['self_k'][:, step : step + 1] = k
            cache['self_v'][:, step : step + 1] = v
            cache['step'] = step + 1
            total = step + 1

            nh = self.self_attention.num_heads
            q_h = mx.unflatten(q, -1, (nh, -1)).transpose(0, 2, 1, 3)
            k_h = mx.unflatten(cache['self_k'][:, :total], -1, (nh, -1)).transpose(0, 2, 1, 3)
            v_h = mx.unflatten(cache['self_v'][:, :total], -1, (nh, -1)).transpose(0, 2, 1, 3)
            scale = math.sqrt(1.0 / q_h.shape[-1])
            x = mx.fast.scaled_dot_product_attention(q_h, k_h, v_h, scale=scale, mask=None)
            x = x.transpose(0, 2, 1, 3).flatten(-2, -1)
            x = self.self_attention.out_proj(x)
        else:
            x = self.self_attention(x, x, x, self_mask)

        x = self.dropout(x)
        x = residual + x
        x = self.self_attention_layer_norm(x)

        # ── Cross-attention (post-LN) with fixed-buffer KV-cache ──
        residual = x

        if cache is not None:
            if cache['cross_k'] is None:
                # First step: compute cross K/V from encoder memory once
                cache['cross_k'] = self.encoder_attention.key_proj(memory)
                cache['cross_v'] = self.encoder_attention.value_proj(memory)
            q = self.encoder_attention.query_proj(x)
            nh = self.encoder_attention.num_heads
            q_h = mx.unflatten(q, -1, (nh, -1)).transpose(0, 2, 1, 3)
            k_h = mx.unflatten(cache['cross_k'], -1, (nh, -1)).transpose(0, 2, 1, 3)
            v_h = mx.unflatten(cache['cross_v'], -1, (nh, -1)).transpose(0, 2, 1, 3)
            scale = math.sqrt(1.0 / q_h.shape[-1])
            x = mx.fast.scaled_dot_product_attention(q_h, k_h, v_h, scale=scale, mask=memory_mask)
            x = x.transpose(0, 2, 1, 3).flatten(-2, -1)
            x = self.encoder_attention.out_proj(x)
        else:
            x = self.encoder_attention(x, memory, memory, memory_mask)

        x = self.dropout(x)
        x = residual + x
        x = self.encoder_attention_layer_norm(x)

        # ── FFN (post-LN) ──
        residual = x
        x = self.fc1(x)
        x = nn.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = residual + x
        x = self.final_layer_norm(x)
        return x


class BiDiDecoder(nn.Module):
    """Decoder stack: embedding + sinusoidal PE + N decoder layers + final LN."""

    def __init__(
        self,
        num_layers: int,
        dims: int,
        num_heads: int,
        mlp_dims: int,
        dropout: float,
        embed_tokens: nn.Embedding,
    ):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.embed_positions = SinusoidalPositionEmbedding(MAX_POSITION_EMBEDDINGS, dims)
        self.layers = [
            BiDiDecoderLayer(dims, num_heads, mlp_dims, dropout) for _ in range(num_layers)
        ]
        self.dropout = nn.Dropout(dropout)
        # Pre-allocated causal masks, reused across decode steps
        self._causal_masks: dict[int, mx.array] = {}

    def __call__(
        self,
        input_ids: mx.array,
        memory: mx.array,
        self_mask: mx.array | None = None,
        memory_mask: mx.array | None = None,
        cache: list[dict] | None = None,
    ) -> mx.array:
        x = self.embed_tokens(input_ids) * EMBED_SCALE

        if cache is not None:
            # Position embedding for the new token only
            step = cache[0]['step']  # tokens already decoded
            full_len = step + input_ids.shape[1]
            x = x + self.embed_positions((1, full_len))[:, -input_ids.shape[1] :, :]
        else:
            x = x + self.embed_positions(x.shape)

        x = self.dropout(x)

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x = layer(x, memory, self_mask, memory_mask, cache=layer_cache)

        return x


class BidiMLXModel(nn.Module):
    """Complete BiDi-eng-pol encoder-decoder model in MLX.

    Usage::

    model = BidiMLXModel()
    model.load_mlx_weights(SAFETENSORS_PATH)
    translated = model.translate(["Hello world", "How are you?"])
    """

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.encoder = BiDiEncoder(
            NUM_ENCODER_LAYERS, D_MODEL, NUM_HEADS, FFN_DIM, DROPOUT, self.embed_tokens
        )
        self.decoder = BiDiDecoder(
            NUM_DECODER_LAYERS, D_MODEL, NUM_HEADS, FFN_DIM, DROPOUT, self.embed_tokens
        )
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=True)

    def encode(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        """Encode source tokens into memory vectors.

        Args:
            input_ids: (batch, src_seq_len) token IDs.
            attention_mask: Optional (batch, 1, 1, src_seq_len) padding mask.

        Returns:
            Memory tensor (batch, src_seq_len, d_model).
        """
        return self.encoder(input_ids, attention_mask)

    def decode(
        self,
        decoder_input_ids: mx.array,
        memory: mx.array,
        self_mask: mx.array | None = None,
        memory_mask: mx.array | None = None,
        cache: list[dict[str, mx.array]] | None = None,
    ) -> mx.array:
        """Decode target tokens given encoder memory.

        Args:
            decoder_input_ids: (batch, tgt_seq_len) token IDs.
            memory: Encoder output (batch, src_seq_len, d_model).
            self_mask: Causal mask for self-attention.
            memory_mask: Padding mask for encoder-decoder attention.
            cache: Optional list of per-layer KV caches for fast autoregressive
                decoding.  When provided, only the *last* token is processed
                through self-attention (the cache provides past K/V).

        Returns:
            Logits tensor (batch, tgt_seq_len, vocab_size).
        """
        hidden = self.decoder(decoder_input_ids, memory, self_mask, memory_mask, cache=cache)
        logits = self.lm_head(hidden)
        return logits

    def __call__(
        self,
        input_ids: mx.array,
        decoder_input_ids: mx.array,
        attention_mask: mx.array | None = None,
        decoder_attention_mask: mx.array | None = None,
    ) -> mx.array:
        """Full forward pass (encode + decode).

        Returns logits of shape (batch, tgt_seq_len, vocab_size).
        """
        memory = self.encode(input_ids, attention_mask)
        return self.decode(
            decoder_input_ids,
            memory,
            self_mask=decoder_attention_mask,
            memory_mask=attention_mask,
        )

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_mlx_weights(self, safetensors_path: str | Path = SAFETENSORS_PATH) -> None:
        """Load Marian-format safetensors weights into this MLX model.

        Handles both FP32 and INT8 quantized model formats:
        - FP32: loads directly from HuggingFace-format weight keys
        - INT8 quantized: calls ``nn.quantize()`` first to build the
          ``QuantizedLinear`` / ``QuantizedEmbedding`` module structure,
          then loads the quantized weights (weight + scales + biases).

        Named load_mlx_weights (not load_weights) to avoid shadowing
        nn.Module.load_weights.
        """
        from safetensors.mlx import load_file

        state = load_file(str(safetensors_path))

        # Detect quantization by looking for 'scales' keys in the state dict
        has_scales = any('scales' in k for k in state)

        if has_scales:
            # Quantized model: build QuantizedLinear/QuantizedEmbedding structure first
            nn.quantize(self, group_size=64, bits=8)

        nested = self._build_nested_weights(state)
        self.update(nested)
        total_mb = sum(v.nbytes for v in state.values()) / 1e6
        logger.info(
            f'Loaded {len(state)} weight tensors from {safetensors_path} ({total_mb:.0f} MB)'
        )

    @staticmethod
    def _build_nested_weights(state: dict[str, mx.array]) -> dict:
        """Build nested weight dict matching MLX module tree from flat weight keys.

        Supports two input formats:
        - HuggingFace Marian format (keys start with ``model.``): e.g.
          ``model.encoder.layers.0.self_attn.q_proj.weight``
        - MLX-native format (our saved model): e.g.
          ``encoder.layers.0.self_attention.query_proj.weight``
          (which may also include ``.scales`` and ``.biases`` for quantized layers).

        Returns a nested dict/list structure matching what MLX's
        ``Module.update()`` expects (not flat dot-separated keys).
        """
        is_hf_format = any(k.startswith('model.') for k in state)

        if is_hf_format:
            return BidiMLXModel._build_from_hf(state)
        else:
            return BidiMLXModel._build_from_mlx(state)

    @staticmethod
    def _build_from_hf(state: dict[str, mx.array]) -> dict:
        """Build nested weights from HuggingFace Marian-format state dict."""
        shared_emb = state.get('model.shared.weight')
        final_bias = state.get('final_logits_bias')

        def _proj_dict(prefix: str, proj_type: str) -> dict:
            result = {}
            for name in ('query', 'key', 'value', 'out'):
                proj_key = f'{name}_proj'
                marian_proj = {'query': 'q', 'key': 'k', 'value': 'v', 'out': 'out'}[name]
                marian_key = f'{prefix}.{proj_type}{marian_proj}_proj'
                d = {}
                w_key = f'{marian_key}.weight'
                if w_key in state:
                    d['weight'] = state[w_key]
                b_key = f'{marian_key}.bias'
                if b_key in state:
                    d['bias'] = state[b_key]
                if d:
                    result[proj_key] = d
            return result

        def _layer_norm(prefix: str, name: str) -> dict:
            d = {}
            w_key = f'{prefix}.{name}.weight'
            b_key = f'{prefix}.{name}.bias'
            if w_key in state:
                d['weight'] = state[w_key]
            if b_key in state:
                d['bias'] = state[b_key]
            return d

        def _ffn(prefix: str) -> dict:
            d = {}
            for name in ('fc1', 'fc2'):
                ffn = {}
                w_key = f'{prefix}.{name}.weight'
                if w_key in state:
                    ffn['weight'] = state[w_key]
                b_key = f'{prefix}.{name}.bias'
                if b_key in state:
                    ffn['bias'] = state[b_key]
                if ffn:
                    d[name] = ffn
            return d

        encoder_layers = []
        for i in range(NUM_ENCODER_LAYERS):
            lp = f'model.encoder.layers.{i}'
            layer = {
                'self_attention': _proj_dict(lp, 'self_attn.'),
                'self_attention_layer_norm': _layer_norm(lp, 'self_attn_layer_norm'),
                **_ffn(lp),
                'final_layer_norm': _layer_norm(lp, 'final_layer_norm'),
            }
            encoder_layers.append(layer)

        decoder_layers = []
        for i in range(NUM_DECODER_LAYERS):
            lp = f'model.decoder.layers.{i}'
            layer = {
                'self_attention': _proj_dict(lp, 'self_attn.'),
                'self_attention_layer_norm': _layer_norm(lp, 'self_attn_layer_norm'),
                'encoder_attention': _proj_dict(lp, 'encoder_attn.'),
                'encoder_attention_layer_norm': _layer_norm(lp, 'encoder_attn_layer_norm'),
                **_ffn(lp),
                'final_layer_norm': _layer_norm(lp, 'final_layer_norm'),
            }
            decoder_layers.append(layer)

        lm_head: dict = {}
        if shared_emb is not None:
            lm_head['weight'] = shared_emb
        if final_bias is not None:
            lm_head['bias'] = final_bias[0]

        return {
            'embed_tokens': {'weight': shared_emb} if shared_emb is not None else {},
            'encoder': {
                'embed_tokens': {'weight': shared_emb} if shared_emb is not None else {},
                'layers': encoder_layers,
            },
            'decoder': {
                'embed_tokens': {'weight': shared_emb} if shared_emb is not None else {},
                'layers': decoder_layers,
            },
            'lm_head': lm_head,
        }

    @staticmethod
    def _build_from_mlx(state: dict[str, mx.array]) -> dict:  # type: ignore[misc]
        """Build nested weights from MLX-native (flat key) format.

        Converts keys like ``encoder.layers.0.self_attention.query_proj.weight``
        into the nested dict/list structure MLX's ``Module.update()`` expects.
        Handles quantized params (``.scales``, ``.biases``) automatically.
        """
        result: dict = {}
        for key, value in sorted(state.items()):
            parts = key.split('.')
            target: dict | list = result
            i = 0
            while i < len(parts):
                part = parts[i]
                is_last = i == len(parts) - 1
                if is_last:
                    if isinstance(target, dict):
                        target[part] = value
                else:
                    next_part = parts[i + 1]
                    try:
                        idx = int(next_part)
                        if isinstance(target, dict):
                            if part not in target:
                                target[part] = []
                            lst = target[part]
                            if isinstance(lst, list):
                                while len(lst) <= idx:
                                    lst.append({})
                                target = lst[idx]
                                i += 1
                    except ValueError:
                        if isinstance(target, dict):
                            if part not in target:
                                target[part] = {}
                            target = target[part]
                i += 1
        return result

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def _load_tokenizer(self) -> None:
        """Load SentencePiece tokenizer for source and target."""
        import sentencepiece as spm

        self._src_sp = spm.SentencePieceProcessor()
        self._src_sp.Load(str(SOURCE_SPM_PATH))
        self._tgt_sp = spm.SentencePieceProcessor()
        self._tgt_sp.Load(str(TARGET_SPM_PATH))

    def tokenize_source(self, texts: list[str]) -> tuple[mx.array, mx.array]:
        """Tokenize source texts (English) and return (input_ids, attention_mask).

        Input texts should NOT include the '>>pol<<' prefix — we add it here
        (token id 5).
        """
        if not hasattr(self, '_src_sp'):
            self._load_tokenizer()

        batch_ids: list[list[int]] = []
        max_len = 0

        for text in texts:
            # Prepend the language token: '>>pol<<' = token_id 5
            lang_token = [5]
            ids = lang_token + self._src_sp.EncodeAsIds(text)
            batch_ids.append(ids)
            max_len = max(max_len, len(ids))

        # Clip to MAX_POSITION_EMBEDDINGS
        max_len = min(max_len, MAX_POSITION_EMBEDDINGS)

        padded = []
        mask = []
        for ids in batch_ids:
            if len(ids) > MAX_POSITION_EMBEDDINGS:
                ids = ids[:MAX_POSITION_EMBEDDINGS]
            pad_len = max_len - len(ids)
            padded.append(ids + [PAD_TOKEN_ID] * pad_len)
            mask.append([1.0] * len(ids) + [0.0] * pad_len)

        return mx.array(padded, dtype=mx.int32), mx.array(mask, dtype=mx.float32)

    def tokenize_target(self, texts: list[str] | None = None, batch_size: int = 1) -> mx.array:
        """Create decoder input IDs starting with pad_token_id (start token).

        During inference we build this autoregressively; this method
        creates the initial (batch, 1) start tensor.
        """
        return mx.full((batch_size, 1), DECODER_START_TOKEN_ID, dtype=mx.int32)

    def decode_target(self, token_ids: mx.array) -> list[str]:
        """Decode target token IDs back to Polish text."""
        if not hasattr(self, '_tgt_sp'):
            self._load_tokenizer()

        # Convert to list-of-lists, handling 1D (single) and 2D (batch)
        raw = token_ids.tolist() if hasattr(token_ids, 'tolist') else token_ids
        if isinstance(raw, (list, tuple)):
            if raw and not isinstance(raw[0], (list, tuple)):
                ids_sequences = [raw]
            else:
                ids_sequences = list(raw)
        else:
            ids_sequences = [[raw]]

        texts = []
        for ids in ids_sequences:
            # ids is always a list here; type checker conservatively disagrees
            stripped = [i for i in ids if i not in (PAD_TOKEN_ID, EOS_TOKEN_ID, 0)]  # type: ignore[arg-type]
            text = self._tgt_sp.DecodeIds(stripped)
            texts.append(text)
        return texts

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        max_new_tokens: int = 128,
        num_beams: int = 1,
    ) -> mx.array:
        """Generate target token IDs using greedy decoding with fixed-buffer KV-cache.

        Uses pre-allocated KV buffers to avoid O(T²) memory from repeated
        ``mx.concatenate`` calls during autoregressive generation.
        """
        batch_size = input_ids.shape[0]
        memory = self.encode(input_ids, attention_mask)

        max_len = 1 + max_new_tokens
        decoder_ids = mx.full((batch_size, max_len), PAD_TOKEN_ID, dtype=mx.int32)
        decoder_ids[:, 0] = DECODER_START_TOKEN_ID
        seq_len = 1

        # Pre-allocate fixed KV-cache buffers per layer
        layer_caches: list[dict] = []
        for _ in range(NUM_DECODER_LAYERS):
            cache = {
                'step': 0,
                'self_k': mx.zeros((batch_size, max_new_tokens, D_MODEL)),
                'self_v': mx.zeros((batch_size, max_new_tokens, D_MODEL)),
                'cross_k': None,  # set on first step
                'cross_v': None,
            }
            layer_caches.append(cache)

        # First step: encode the full source once for cross-attention cache
        cache_prompt = mx.full((batch_size, 1), DECODER_START_TOKEN_ID, dtype=mx.int32)
        logits = self.decode(cache_prompt, memory, cache=layer_caches)

        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        decoder_ids[:, 1] = next_token.astype(mx.int32)
        seq_len = 2

        if mx.all(mx.equal(next_token, EOS_TOKEN_ID)):
            return decoder_ids[:, :seq_len]

        for _ in range(1, max_new_tokens):
            new_token_ids = mx.reshape(decoder_ids[:, seq_len - 1], (-1, 1))

            logits = self.decode(new_token_ids, memory, cache=layer_caches)
            next_token_logits = logits[:, -1, :]
            if self.lm_head.bias is not None:
                next_token_logits = next_token_logits + self.lm_head.bias

            next_token = mx.argmax(next_token_logits, axis=-1)
            decoder_ids[:, seq_len] = next_token.astype(mx.int32)
            seq_len += 1

            if mx.all(mx.equal(next_token, EOS_TOKEN_ID)):
                break

        return decoder_ids[:, :seq_len]

    def translate(
        self,
        texts: list[str],
        max_new_tokens: int = 128,
        progress_callback: Callable | None = None,
        batch_size: int = 4,
    ) -> list[str]:
        """Translate a list of English texts to Polish.

        Uses fixed-buffer KV-cache for O(T) decoding with stable memory.
        Default batch_size=4 (optimal on M1 8GB; larger batches add padding
        overhead and memory pressure).
        """
        if not texts:
            return []

        total = len(texts)
        results: list[str] = [''] * total
        start_time = time.time()

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_texts = texts[batch_start:batch_end]
            b = len(batch_texts)

            input_ids, attention_mask = self.tokenize_source(batch_texts)
            memory = self.encode(input_ids, create_padding_mask(attention_mask))

            max_len = 1 + max_new_tokens
            decoder_ids = mx.full((b, max_len), PAD_TOKEN_ID, dtype=mx.int32)
            decoder_ids[:, 0] = DECODER_START_TOKEN_ID
            seq_len = 1

            # Pre-allocate fixed KV-cache (static, no mx.concatenate in loop)
            layer_caches: list[dict] = []
            for _ in range(NUM_DECODER_LAYERS):
                layer_caches.append(
                    {
                        'step': 0,
                        'self_k': mx.zeros((b, max_new_tokens, D_MODEL)),
                        'self_v': mx.zeros((b, max_new_tokens, D_MODEL)),
                        'cross_k': None,
                        'cross_v': None,
                    }
                )

            # First step: decode start token (initialises cross-attn cache)
            start_input = decoder_ids[:, 0:1]
            logits = self.decode(start_input, memory, cache=layer_caches)
            next_token = mx.argmax(logits[:, -1, :], axis=-1)
            decoder_ids[:, 1] = next_token.astype(mx.int32)
            seq_len = 2

            if mx.all(mx.equal(next_token, EOS_TOKEN_ID)):
                batch_results = self.decode_target(decoder_ids[:, :seq_len])
                for j, text in enumerate(batch_results):
                    results[batch_start + j] = text
                if progress_callback:
                    elapsed = time.time() - start_time
                    progress_callback(batch_end, total, batch_end / elapsed if elapsed > 0 else 0)
                continue

            # Remaining steps
            for _ in range(1, max_new_tokens):
                new_ids = mx.reshape(decoder_ids[:, seq_len - 1], (-1, 1))
                logits = self.decode(new_ids, memory, cache=layer_caches)
                next_token_logits = logits[:, -1, :]
                if self.lm_head.bias is not None:
                    next_token_logits = next_token_logits + self.lm_head.bias
                next_token = mx.argmax(next_token_logits, axis=-1)

                decoder_ids[:, seq_len] = next_token.astype(mx.int32)
                seq_len += 1

                if mx.all(mx.equal(next_token, EOS_TOKEN_ID)):
                    break

            batch_results = self.decode_target(decoder_ids[:, :seq_len])
            for j, text in enumerate(batch_results):
                results[batch_start + j] = text

            # Force synchronisation and GC to keep peak memory low
            mx.eval(next_token)
            for c in layer_caches:
                c['self_k'] = None
                c['self_v'] = None
                c['cross_k'] = None
                c['cross_v'] = None

            if progress_callback:
                elapsed = time.time() - start_time
                rate = batch_end / elapsed if elapsed > 0 else 0
                progress_callback(batch_end, total, rate)

        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_causal_mask(seq_len: int) -> mx.array:
    """Create additive causal attention mask for decoder self-attention.

    Returns (1, 1, seq_len, seq_len) mask with 0 for allowed positions
    and -inf for masked positions.
    """
    mask = mx.full((seq_len, seq_len), -mx.inf)
    mask = mx.triu(mask, k=1)  # Upper triangle (j > i) = -inf
    # Diagonal and below (j <= i) = 0
    mask = mask + mx.eye(seq_len) * 0.0
    return mask[None, None, :, :]  # (1, 1, seq_len, seq_len)


def create_padding_mask(attention_mask: mx.array) -> mx.array:
    """Convert token-level attention mask (batch, seq_len) to 4D additive mask.

    Returns (batch, 1, 1, seq_len) with 0 for valid positions, -inf for padding.
    Uses mx.where to avoid NaN from 0 * -inf in IEEE 754.
    """
    # attention_mask: 1.0 = valid, 0.0 = padding
    mask = mx.where(attention_mask == 0.0, -mx.inf, 0.0)
    return mask[:, None, None, :]


# ---------------------------------------------------------------------------
# Convenience: check if MLX backend is available
# ---------------------------------------------------------------------------


def is_available() -> bool:
    """Check if MLX is available (Apple Silicon) and model files exist."""
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        return False
    return SAFETENSORS_PATH.exists()
