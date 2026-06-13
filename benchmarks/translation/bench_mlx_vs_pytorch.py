#!/usr/bin/env python3
"""Benchmark comparing PyTorch (allegro) vs MLX INT8 translation performance.

Each model is loaded in an isolated subprocess to avoid memory conflicts.
Measures:
- Sentences per second (throughput)
- Peak memory usage (via tracemalloc / psutil)
- Translation quality comparison (sample)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def extract_dialogue_texts(ass_path: str) -> list[str]:
    """Extract dialogue text lines from an ASS file."""
    texts = []
    in_events = False
    with open(ass_path) as f:
        for line in f:
            if line.startswith('[Events]'):
                in_events = True
                continue
            if in_events and line.startswith('Dialogue:'):
                parts = line.split(',', 9)
                if len(parts) >= 10:
                    text = parts[9]
                    text = re.sub(r'\{.*?\}', '', text)
                    text = text.replace('\\N', ' ').replace('\\n', ' ').strip()
                    if text and not text.startswith('{'):
                        texts.append(text)
    return texts


def run_bench_subprocess(backend: str, texts: list[str], batch_size: int) -> dict:
    """Run a benchmark in an isolated subprocess and return results as dict."""
    import json as _json

    script = f'''
import sys, time, gc, os
sys.path.insert(0, {str(REPO_ROOT)!r})

texts = {_json.dumps(texts)}
batch_size = {batch_size}

# Memory tracking (resident size in MB)
def get_rss():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        return 0

peak_rss = [get_rss()]

t0 = time.time()

if "{backend}" == "mlx":
    from movie_translator.translation.mlx_backend import BidiMLXModel
    model = BidiMLXModel()
    model.load_mlx_weights()
    load_time = time.time() - t0
    peak_rss[0] = max(peak_rss[0], get_rss())

    t1 = time.time()
    translated = model.translate(texts, max_new_tokens=128, batch_size=batch_size)
    translate_time = time.time() - t1
    peak_rss[0] = max(peak_rss[0], get_rss())

    # Model size
    params = model.parameters()
    def count_size(d):
        import mlx.core as mx
        total = 0
        if isinstance(d, dict):
            for v in d.values():
                if isinstance(v, mx.array):
                    total += v.nbytes
                else:
                    total += count_size(v)
        elif isinstance(d, list):
            for v in d:
                total += count_size(v)
        return total
    model_mb = count_size(params) / (1024*1024)

elif "{backend}" == "pytorch":
    import torch
    from transformers import MarianMTModel, MarianTokenizer
    from movie_translator.translation.enhancements import postprocess_translation

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    tokenizer = MarianTokenizer.from_pretrained("allegro/BiDi-eng-pol")
    model = MarianMTModel.from_pretrained(
        "allegro/BiDi-eng-pol",
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    load_time = time.time() - t0
    peak_rss[0] = max(peak_rss[0], get_rss())

    # Warm up
    _ = model.generate(
        **tokenizer([">>pol<< Hello world"], return_tensors="pt", padding=True).to(device),
        max_new_tokens=10,
    )

    t1 = time.time()
    translated = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        prefixed = [f">>pol<< {{t}}" for t in batch]
        inputs = tokenizer(prefixed, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=128, num_beams=1, do_sample=False)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translated.extend(decoded)

        if i > 0 and i % (batch_size * 5) == 0:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()
        peak_rss[0] = max(peak_rss[0], get_rss())

    translate_time = time.time() - t1

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

    translated = [postprocess_translation(t) for t in translated]
    model_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024*1024)
    device = device
else:
    raise ValueError(f"Unknown backend: {{backend}}")

total = len(texts)
elapsed = translate_time
result = {{
    "backend": "{backend}",
    "total_sentences": total,
    "load_time_s": round(load_time, 2),
    "translate_time_s": round(elapsed, 2),
    "sentences_per_sec": round(total / elapsed, 1) if elapsed > 0 else 0,
    "model_size_mb": round(model_mb, 1),
    "peak_rss_mb": round(peak_rss[0], 1),
    "translated": translated,
}}
print(_json.dumps(result))
'''
    result = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        stderr = result.stderr[:500]
        return {'error': f'Subprocess failed: {stderr}', 'stderr': result.stderr[:2000]}
    try:
        return json.loads(result.stdout.strip().split('\n')[-1])
    except (json.JSONDecodeError, ValueError) as e:
        return {'error': f'JSON parse failed: {e}', 'stdout': result.stdout[-500:]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-size', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--backend', choices=['mlx', 'pytorch', 'both'], default='both')
    args = parser.parse_args()

    ass_path = REPO_ROOT / 'benchmarks' / 'asr' / 'refs' / 'en_ref.ass'
    texts = extract_dialogue_texts(str(ass_path))
    texts = [t for t in texts if 3 <= len(t) <= 300]
    print(f'Extracted {len(texts)} dialogue lines from {ass_path.name}')

    # Quality sample (small, for comparison)
    quality_sample = texts[: args.sample_size]

    results = {}

    for backend in ['mlx', 'pytorch'] if args.backend == 'both' else [args.backend]:
        print(
            f'\nBenchmarking {backend.upper()} ({len(quality_sample)} lines, '
            f'batch={args.batch_size})...'
        )
        sys.stdout.flush()

        r = run_bench_subprocess(backend, quality_sample, args.batch_size)

        if 'error' in r:
            print(f'  ERROR: {r["error"]}')
            results[backend] = r
            continue

        results[backend] = r
        print(f'  Load:       {r["load_time_s"]:.1f}s')
        print(f'  Translate:  {r["translate_time_s"]:.1f}s')
        print(f'  Throughput: {r["sentences_per_sec"]:.1f} sentences/s')
        print(f'  Model size: {r["model_size_mb"]:.0f} MB')
        print(f'  Peak RSS:   {r["peak_rss_mb"]:.0f} MB')

    # Summary
    print('\n' + '=' * 65)
    print('BENCHMARK SUMMARY')
    print('=' * 65)

    if 'mlx' in results and 'error' not in results['mlx']:
        mlx = results['mlx']
        print('\nMLX INT8 (Metal GPU):')
        print(f'  {mlx["total_sentences"]} sentences in {mlx["translate_time_s"]:.1f}s')
        print(f'  {mlx["sentences_per_sec"]:.1f} sentences/sec')
        print(f'  Model: {mlx["model_size_mb"]:.0f} MB, Peak RSS: {mlx["peak_rss_mb"]:.0f} MB')

    if 'pytorch' in results and 'error' not in results['pytorch']:
        pt = results['pytorch']
        print('\nPyTorch MPS:')
        print(f'  {pt["total_sentences"]} sentences in {pt["translate_time_s"]:.1f}s')
        print(f'  {pt["sentences_per_sec"]:.1f} sentences/sec')
        print(f'  Model: {pt["model_size_mb"]:.0f} MB, Peak RSS: {pt["peak_rss_mb"]:.0f} MB')

    if 'mlx' in results and 'pytorch' in results:
        if 'error' not in results['mlx'] and 'error' not in results['pytorch']:
            m, p = results['mlx'], results['pytorch']
            speed_ratio = p['sentences_per_sec'] / m['sentences_per_sec']
            size_ratio = p['model_size_mb'] / m['model_size_mb']
            mem_ratio = p['peak_rss_mb'] / m['peak_rss_mb']

            print('\nMLX vs PyTorch:')
            print(
                f'  Speed:    {speed_ratio:.1f}x'
                + (' (MLX faster)' if speed_ratio < 1 else ' (PyTorch faster)')
            )
            print(f'  Size:     {size_ratio:.1f}x smaller (MLX)')
            print(f'  Memory:   {mem_ratio:.1f}x less (MLX)')

    # Quality comparison
    if 'mlx' in results and 'pytorch' in results:
        if 'error' not in results['mlx'] and 'error' not in results['pytorch']:
            print('\n\nTranslation Quality (first 10):')
            print(f'{"#":3s} {"Source":35s} {"PyTorch":30s} {"MLX INT8":30s}')
            print('-' * 100)
            for i in range(min(10, len(quality_sample))):
                src = (
                    quality_sample[i][:33] + '..'
                    if len(quality_sample[i]) > 33
                    else quality_sample[i]
                )
                pt_t = (
                    results['pytorch']['translated'][i][:28] + '..'
                    if len(results['pytorch']['translated'][i]) > 28
                    else results['pytorch']['translated'][i]
                )
                mlx_t = (
                    results['mlx']['translated'][i][:28] + '..'
                    if len(results['mlx']['translated'][i]) > 28
                    else results['mlx']['translated'][i]
                )
                print(f'{i:3d} {src:35s} {pt_t:30s} {mlx_t:30s}')


if __name__ == '__main__':
    main()
