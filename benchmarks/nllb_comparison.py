#!/usr/bin/env python3
"""Benchmark: Compare Allegro BiDi, NLLB-600M, NLLB-1.3B, and Apple Translation.

Measures translation quality (side-by-side output), speed, and memory for each backend.

Usage:
    python -m benchmarks.nllb_comparison
    python -m benchmarks.nllb_comparison --models allegro nllb-600m
    python -m benchmarks.nllb_comparison --srt path/to/file.srt
"""

import argparse
import gc
import json
import resource
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).parent.parent

# Sample subtitle lines for benchmarking (from Oppenheimer + anime-style dialogue)
DEFAULT_LINES = [
    # Oppenheimer (short, dramatic)
    'We knew the world would not be the same.',
    'Few people laughed.',
    'Few people cried.',
    'Most people were silent.',
    'I remembered the line from the Hindu scripture, the Bhagavad Gita.',
    'Now, I am become death, the destroyer of worlds.',
    'I suppose we all thought that one way or another.',
    # Anime dialogue (typical subtitle patterns)
    "What's wrong? You look pale.",
    "I won't let you get away with this!",
    "Everyone, fall back! It's too dangerous!",
    'This power... it was inside me all along?',
    "Don't underestimate me!",
    'We have no choice but to fight.',
    'Thank you... for everything.',
    "Is that so? Then I'll show you what real strength looks like.",
    'The enemy is approaching from the north!',
    "I'm not the same person I used to be.",
    'Damn it! We were too late!',
    'Promise me you will come back alive.',
    "This isn't over yet!",
    # Longer/complex sentences
    'If we do not act now, everything we have fought for will be lost.',
    'The Band of the Hawk will be the vanguard of the assault.',
    "I've been searching for you for a very long time.",
    'In this world, is the destiny of mankind controlled by some transcendental entity or law?',
    "It doesn't matter what happens to me, as long as you're safe.",
    # Short interjections (stress-test for small inputs)
    'Run!',
    'Watch out!',
    'Impossible...',
    'March!',
]


def parse_srt(path: Path) -> list[str]:
    """Extract text lines from an SRT file."""
    lines = []
    current_text_parts: list[str] = []
    for raw_line in path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line:
            if current_text_parts:
                lines.append(' '.join(current_text_parts))
                current_text_parts = []
        elif '-->' in line or line.isdigit():
            continue
        else:
            current_text_parts.append(line)
    if current_text_parts:
        lines.append(' '.join(current_text_parts))
    return lines


def get_memory_mb() -> float:
    """Get current RSS memory in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def translate_per_line(translator, texts: list[str]) -> list[str]:
    """Translate each line individually, bypassing the sentence merger."""
    results = []
    for i in range(0, len(texts), translator.batch_size):
        batch = texts[i : i + translator.batch_size]
        processed = translator._preprocess_texts(batch)
        encoded = translator._encode_texts(processed)
        outputs = translator._generate_translations(encoded)
        decoded = translator._decode_outputs(outputs)
        results.extend(decoded)
    return results


def benchmark_hf_model(
    model_key: str, texts: list[str], device: str, use_merger: bool = False
) -> dict:
    """Benchmark a HuggingFace-based translation model."""
    from movie_translator.translation.translator import SubtitleTranslator

    gc.collect()
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    mem_before = get_memory_mb()

    # Load
    t0 = time.perf_counter()
    translator = SubtitleTranslator(model_key=model_key, device=device, enable_enhancements=False)
    load_ok = translator.load_model()
    load_time = time.perf_counter() - t0

    if not load_ok:
        return {'model': model_key, 'error': 'Failed to load model'}

    mem_after_load = get_memory_mb()

    # Translate
    t0 = time.perf_counter()
    if use_merger:
        translations = translator.translate_texts(texts)
    else:
        translations = translate_per_line(translator, texts)
    translate_time = time.perf_counter() - t0

    mem_after_translate = get_memory_mb()

    # Model size on disk
    model_size_mb = 0.0
    if translator.model is not None:
        param_bytes = sum(p.nelement() * p.element_size() for p in translator.model.parameters())
        model_size_mb = param_bytes / (1024 * 1024)

    result = {
        'model': model_key,
        'huggingface_id': translator.model_config.get('huggingface_id', model_key),
        'load_time_s': round(load_time, 2),
        'translate_time_s': round(translate_time, 2),
        'lines': len(texts),
        'lines_per_second': round(len(texts) / translate_time, 1) if translate_time > 0 else 0,
        'model_memory_mb': round(model_size_mb, 1),
        'rss_before_mb': round(mem_before, 1),
        'rss_after_load_mb': round(mem_after_load, 1),
        'rss_after_translate_mb': round(mem_after_translate, 1),
        'translations': translations,
    }

    translator.cleanup()
    gc.collect()
    return result


def benchmark_apple(texts: list[str], use_merger: bool = False) -> dict | None:
    """Benchmark Apple Translation backend."""
    try:
        from movie_translator.translation.apple_backend import AppleTranslationBackend, is_available

        if not is_available():
            return None
    except Exception:
        return None

    gc.collect()
    mem_before = get_memory_mb()

    t0 = time.perf_counter()
    backend = AppleTranslationBackend(batch_size=200, enable_enhancements=False)
    load_time = time.perf_counter() - t0

    mem_after_load = get_memory_mb()

    t0 = time.perf_counter()
    # Apple backend always translates per-line (no sentence merger internally)
    translations = backend.translate_texts(texts)
    translate_time = time.perf_counter() - t0

    mem_after_translate = get_memory_mb()

    result = {
        'model': 'apple',
        'huggingface_id': 'Apple Translation Framework (macOS)',
        'load_time_s': round(load_time, 2),
        'translate_time_s': round(translate_time, 2),
        'lines': len(texts),
        'lines_per_second': round(len(texts) / translate_time, 1) if translate_time > 0 else 0,
        'model_memory_mb': 0,
        'rss_before_mb': round(mem_before, 1),
        'rss_after_load_mb': round(mem_after_load, 1),
        'rss_after_translate_mb': round(mem_after_translate, 1),
        'translations': translations,
    }

    backend.cleanup()
    gc.collect()
    return result


def print_results(results: list[dict], texts: list[str]) -> None:
    """Print benchmark results as tables."""
    print('\n' + '=' * 90)
    print('PERFORMANCE COMPARISON')
    print('=' * 90)

    # Header
    header = f'{"Model":<20} {"Load (s)":<10} {"Translate (s)":<14} {"Lines/s":<10} {"Model MB":<10} {"RSS delta MB":<12}'
    print(header)
    print('-' * 90)

    for r in results:
        if 'error' in r:
            print(f'{r["model"]:<20} ERROR: {r["error"]}')
            continue
        rss_delta = r['rss_after_load_mb'] - r['rss_before_mb']
        print(
            f'{r["model"]:<20} {r["load_time_s"]:<10.2f} {r["translate_time_s"]:<14.2f} '
            f'{r["lines_per_second"]:<10.1f} {r["model_memory_mb"]:<10.1f} {rss_delta:<12.1f}'
        )

    # Side-by-side translations
    print('\n' + '=' * 90)
    print('TRANSLATION QUALITY COMPARISON (side-by-side)')
    print('=' * 90)

    # Show a representative subset
    sample_indices = list(range(min(len(texts), 30)))

    for i in sample_indices:
        print(f'\n--- Line {i + 1} ---')
        print(f'  EN: {texts[i]}')
        for r in results:
            if 'error' in r:
                continue
            translation = r['translations'][i] if i < len(r['translations']) else '(missing)'
            print(f'  {r["model"]:<14}: {translation}')


def main():
    parser = argparse.ArgumentParser(description='Benchmark translation models')
    parser.add_argument(
        '--models',
        nargs='+',
        default=['allegro', 'nllb-600m', 'nllb-1.3b', 'apple'],
        help='Models to benchmark',
    )
    parser.add_argument('--srt', type=Path, help='SRT file to use as input')
    parser.add_argument('--device', default='mps', choices=['mps', 'cpu'], help='Torch device')
    parser.add_argument(
        '--no-merge',
        action='store_true',
        help='Bypass sentence merger for clean per-line comparison',
    )
    parser.add_argument('--save', action='store_true', help='Save results to JSON')
    args = parser.parse_args()

    # Collect input texts
    if args.srt:
        texts = parse_srt(args.srt)
        print(f'Loaded {len(texts)} lines from {args.srt}')
    else:
        texts = DEFAULT_LINES
        print(f'Using {len(texts)} built-in benchmark lines')

    use_merger = not args.no_merge
    print(f'Device: {args.device}')
    print(f'Sentence merger: {"ON" if use_merger else "OFF (per-line)"}')
    print(f'Models: {", ".join(args.models)}')

    results = []

    for model_key in args.models:
        if model_key == 'apple':
            print('\n>>> Benchmarking Apple Translation...')
            r = benchmark_apple(texts, use_merger=use_merger)
            if r is None:
                print('  Apple Translation not available on this system')
                continue
        else:
            print(f'\n>>> Benchmarking {model_key}...')
            r = benchmark_hf_model(model_key, texts, args.device, use_merger=use_merger)

        if 'error' not in r:
            print(
                f'  Load: {r["load_time_s"]}s | Translate: {r["translate_time_s"]}s | {r["lines_per_second"]} lines/s'
            )
        else:
            print(f'  ERROR: {r["error"]}')

        results.append(r)

    print_results(results, texts)

    if args.save:
        out_path = (
            REPO_ROOT
            / 'benchmarks'
            / 'results'
            / f'nllb_comparison_{datetime.now(UTC).strftime("%Y-%m-%d_%H%M")}.json'
        )
        save_data = {
            'timestamp': datetime.now(UTC).isoformat(),
            'device': args.device,
            'input_lines': len(texts),
            'results': [{k: v for k, v in r.items() if k != 'translations'} for r in results],
            'translations': {
                r['model']: r.get('translations', []) for r in results if 'error' not in r
            },
        }
        out_path.write_text(json.dumps(save_data, indent=2, ensure_ascii=False))
        print(f'\nResults saved to {out_path}')


if __name__ == '__main__':
    main()
