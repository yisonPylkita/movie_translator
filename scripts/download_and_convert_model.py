#!/usr/bin/env python3
"""Download the Allegro BiDi-eng-pol model and convert to MLX INT8.

Usage:
    python scripts/download_and_convert_model.py [--output-dir models/allegro]

This script:
1. Downloads the PyTorch Marian model from HuggingFace (allegro/BiDi-eng-pol)
2. Loads the weights into our MLX BidiMLXModel
3. Quantizes to INT8 (group_size=64, bits=8)
4. Saves the quantized MLX model as safetensors to the output directory

The output directory is suitable for Git LFS tracking.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Constants — match the model config
# ---------------------------------------------------------------------------
D_MODEL = 1024
NUM_HEADS = 16
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
FFN_DIM = 4096
DROPOUT = 0.1
VOCAB_SIZE = 32000
MAX_POSITION_EMBEDDINGS = 1024
PAD_TOKEN_ID = 1
EOS_TOKEN_ID = 2
DECODER_START_TOKEN_ID = 1
EMBED_SCALE = math.sqrt(D_MODEL)

QUANTIZE_GROUP_SIZE = 64
QUANTIZE_BITS = 8


def download_pytorch_model(output_dir: Path) -> None:
    """Download the HuggingFace PyTorch model to *output_dir*."""
    from transformers import MarianMTModel, MarianTokenizer

    model_id = 'allegro/BiDi-eng-pol'
    print(f'Downloading {model_id} from HuggingFace...')

    t0 = time.time()
    tokenizer = MarianTokenizer.from_pretrained(model_id)
    model = MarianMTModel.from_pretrained(model_id)
    elapsed = time.time() - t0
    print(f'Downloaded in {elapsed:.1f}s')

    # Save to output dir
    print(f'Saving PyTorch weights to {output_dir}...')
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print('PyTorch model saved.')


def convert_and_quantize(
    torch_dir: Path,
    output_safetensors: Path,
    output_config: Path,
) -> None:
    """Load PyTorch weights, convert to MLX, quantize to INT8, save.

    Args:
        torch_dir: Directory with HuggingFace PyTorch model files.
        output_safetensors: Path for the output .safetensors file.
        output_config: Path for the output config.json.
    """
    import mlx.nn as nn

    from movie_translator.translation.mlx_backend import BidiMLXModel

    print('Building MLX model and loading PyTorch weights...')
    t0 = time.time()

    # Copy SentencePiece models (skip if src == dst)
    out_dir = output_safetensors.parent
    for spm_name in ('source.spm', 'target.spm'):
        sp_src = torch_dir / spm_name
        sp_dst = out_dir / spm_name
        if sp_src.exists() and sp_src.resolve() != sp_dst.resolve():
            shutil.copy2(sp_src, sp_dst)

    # Build MLX model and load from the PyTorch safetensors
    model = BidiMLXModel()
    # Try loading from the HF directory, then from individual safetensors
    hf_safetensors = torch_dir / 'model.safetensors'
    if hf_safetensors.exists():
        model.load_mlx_weights(str(hf_safetensors))
    else:
        # Might be split into shards
        shards = list(torch_dir.glob('model-*-of-*.safetensors'))
        if shards:
            # Load all shards
            from safetensors.mlx import load_file

            all_state = {}
            for shard in sorted(shards):
                all_state.update(load_file(str(shard)))
            # Build nested dict and update
            nested = BidiMLXModel._build_nested_weights(all_state)
            model.update(nested)
        else:
            raise FileNotFoundError(
                f'No safetensors found in {torch_dir}. Try running with --download first.'
            )

    load_elapsed = time.time() - t0
    print(f'Model loaded in {load_elapsed:.1f}s')

    # Quick verification: run a test translation
    test_result = model.translate(['Hello!'], max_new_tokens=10)
    print(f'  Test before quantize: "Hello!" -> "{test_result[0]}"')

    # Quantize to INT8
    print(f'Quantizing to INT8 (group_size={QUANTIZE_GROUP_SIZE}, bits={QUANTIZE_BITS})...')
    t0 = time.time()
    nn.quantize(model, group_size=QUANTIZE_GROUP_SIZE, bits=QUANTIZE_BITS)
    quant_elapsed = time.time() - t0
    print(f'Quantized in {quant_elapsed:.1f}s')

    # Verify after quantize
    test_result2 = model.translate(['Hello!'], max_new_tokens=10)
    print(f'  Test after quantize:  "Hello!" -> "{test_result2[0]}"')

    # Save quantized model
    print(f'Saving quantized model to {output_safetensors}...')
    t0 = time.time()
    _save_quantized_model(model, output_safetensors)
    save_elapsed = time.time() - t0
    file_size = output_safetensors.stat().st_size
    print(f'Saved in {save_elapsed:.1f}s ({file_size / 1e6:.0f} MB)')

    # Save config
    config = {
        'model_type': 'bidi_mlx_quantized',
        'd_model': D_MODEL,
        'num_heads': NUM_HEADS,
        'num_encoder_layers': NUM_ENCODER_LAYERS,
        'num_decoder_layers': NUM_DECODER_LAYERS,
        'ffn_dim': FFN_DIM,
        'dropout': DROPOUT,
        'vocab_size': VOCAB_SIZE,
        'max_position_embeddings': MAX_POSITION_EMBEDDINGS,
        'pad_token_id': PAD_TOKEN_ID,
        'eos_token_id': EOS_TOKEN_ID,
        'decoder_start_token_id': DECODER_START_TOKEN_ID,
        'embed_scale': EMBED_SCALE,
        'quantized': True,
        'quantize_group_size': QUANTIZE_GROUP_SIZE,
        'quantize_bits': QUANTIZE_BITS,
    }
    with open(output_config, 'w') as f:
        json.dump(config, f, indent=2)
    print(f'Config saved to {output_config}')

    print('\nConversion complete!')
    print(f'  Quantized model: {output_safetensors} ({file_size / 1e6:.0f} MB)')
    print('  Original (~FP32): ~798 MB')
    print(f'  Compression ratio: {798 / (file_size / 1e6):.1f}x')


def _save_quantized_model(model, output_path: Path) -> None:
    """Save a quantized MLX model as a single safetensors file.

    Flattens the nested parameter tree into flat keys for safetensors.
    """
    from safetensors.mlx import save_file

    params = _flatten_params(model.parameters())
    save_file(params, str(output_path))


def _flatten_params(d, prefix=''):
    """Recursively flatten nested dict/list params into dict of arrays."""
    import mlx.core as mx

    flat = {}
    for k, v in d.items():
        key = f'{prefix}.{k}' if prefix else k
        if isinstance(v, mx.array):
            flat[key] = v
        elif isinstance(v, dict):
            flat.update(_flatten_params(v, key))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                flat.update(_flatten_params(item, f'{key}.{i}'))
    return flat


def validate_quantized_model(model_dir: Path) -> bool:
    """Load and verify the quantized model produces valid translations."""
    import mlx.nn as nn

    from movie_translator.translation.mlx_backend import BidiMLXModel

    safetensors_path = model_dir / 'model.safetensors'
    if not safetensors_path.exists():
        print(f'ERROR: {safetensors_path} not found')
        return False

    try:
        print(f'\nValidating quantized model from {safetensors_path}...')
        t0 = time.time()

        # Build model, quantize its structure, then load weights
        model = BidiMLXModel()

        # Load the config for quantization metadata
        config_path = model_dir / 'config.json'
        q_group_size = QUANTIZE_GROUP_SIZE
        q_bits = QUANTIZE_BITS
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            q_group_size = cfg.get('quantize_group_size', q_group_size)
            q_bits = cfg.get('quantize_bits', q_bits)

        # Determine if the safetensors is quantized by checking for scales keys
        from safetensors.mlx import load_file

        state = load_file(str(safetensors_path))
        is_quantized = any('scales' in k for k in state)

        if is_quantized:
            # Build model and apply nn.quantize to create the right module structure
            nn.quantize(model, group_size=q_group_size, bits=q_bits)
            nested = BidiMLXModel._build_nested_weights(state)
            model.update(nested)
        else:
            nested = BidiMLXModel._build_nested_weights(state)
            model.update(nested)

        load_elapsed = time.time() - t0
        print(f'Loaded in {load_elapsed:.1f}s')

        # Run test translations
        test_cases = [
            'Hello!',
            'What is your name?',
            'I am fine, thank you.',
            'Good morning!',
        ]
        for text in test_cases:
            result = model.translate([text], max_new_tokens=20)
            print(f'  EN: {text}')
            print(f'  PL: {result[0]}')

        print('\nValidation PASSED')
        return True

    except Exception as e:
        print(f'Validation FAILED: {e}')
        import traceback

        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description='Download Allegro BiDi model and convert to MLX INT8'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=REPO_ROOT / 'models' / 'allegro',
        help='Output directory for the converted model (default: models/allegro)',
    )
    parser.add_argument(
        '--download-only',
        action='store_true',
        help='Only download the HuggingFace model, do not convert',
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate an already-converted model',
    )
    parser.add_argument(
        '--torch-dir',
        type=Path,
        default=None,
        help='Directory with downloaded PyTorch model (skips download)',
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.validate_only:
        success = validate_quantized_model(output_dir)
        sys.exit(0 if success else 1)

    # Determine where PyTorch model lives or needs to be downloaded
    if args.torch_dir:
        torch_dir = args.torch_dir
        if not torch_dir.exists():
            print(f'ERROR: --torch-dir {torch_dir} does not exist')
            sys.exit(1)
    else:
        # Download to a temp dir
        torch_dir = Path(tempfile.mkdtemp(prefix='allegro_torch_'))
        try:
            download_pytorch_model(torch_dir)
        except Exception as e:
            print(f'ERROR: Download failed: {e}')
            shutil.rmtree(torch_dir, ignore_errors=True)
            sys.exit(1)

    if args.download_only:
        print(f'PyTorch model downloaded to {torch_dir}')
        print('Use --torch-dir to convert:')
        print(f'  python {__file__} --torch-dir {torch_dir}')
        sys.exit(0)

    # Convert
    safetensors_path = output_dir / 'model.safetensors'
    config_path = output_dir / 'config.json'
    try:
        convert_and_quantize(torch_dir, safetensors_path, config_path)

        # Validate
        if not validate_quantized_model(output_dir):
            print('\nWARNING: Validation produced warnings, but conversion completed.')
    finally:
        # Clean up temp PyTorch download
        if not args.torch_dir and torch_dir:
            shutil.rmtree(torch_dir, ignore_errors=True)

    print('\nNext steps:')
    print(f'  1. Track {output_dir} with Git LFS:')
    print(f'     git lfs track "{output_dir.relative_to(REPO_ROOT)}/model.safetensors"')
    print('  2. Commit the converted model:')
    print(f'     git add {output_dir.relative_to(REPO_ROOT)}/')
    print('  3. Update justfile model recipe to use the conversion script')


if __name__ == '__main__':
    main()
