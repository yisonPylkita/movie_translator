"""Bake-off orchestrator. Runs each engine adapter in its OWN venv, one at a
time (the one-GPU rule: Metal/MPS/CPU inference must not overlap). Resilient —
a failing config writes ok:false and the run continues. Resumable — a config
whose result JSON already exists with ok:true is skipped.

    python3 run.py --variant seg                 # full segment matrix
    python3 run.py --variant full --engines mlx faster --langs ja
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / 'results'

# engine -> (env name, adapter file, models, languages)
ENGINES = {
    'openai': ('openai', 'openai_adapter.py', ['small', 'large-v3'], ['en', 'ja']),
    'faster': ('faster', 'faster_adapter.py', ['small', 'large-v3'], ['en', 'ja']),
    'mlx': ('mlx', 'mlx_adapter.py', ['small', 'large-v3'], ['en', 'ja']),
    'whisperx': ('whisperx', 'whisperx_adapter.py', ['small', 'large-v3'], ['en', 'ja']),
    'whispercpp': ('whispercpp', 'whispercpp_adapter.py', ['small', 'large-v3'], ['en', 'ja']),
    'kotoba': ('kotoba', 'kotoba_adapter.py', ['v2.0'], ['ja']),
}


def env_python(env: str) -> Path:
    return HERE / 'envs' / env / 'bin' / 'python'


def already_ok(out: Path) -> bool:
    if not out.is_file():
        return False
    try:
        import json

        return bool(json.loads(out.read_text())['meta']['ok'])
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=['seg', 'full'], default='seg')
    ap.add_argument('--engines', nargs='*', default=list(ENGINES))
    ap.add_argument('--langs', nargs='*', default=None)
    ap.add_argument('--models', nargs='*', default=None)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    jobs = []
    for eng in args.engines:
        envname, adapter, models, langs = ENGINES[eng]
        for model in args.models or models:
            if model not in models:
                continue
            for lang in args.langs or langs:
                if lang not in langs:
                    continue
                jobs.append((eng, envname, adapter, model, lang))

    print(f'{len(jobs)} configs, variant={args.variant}\n', flush=True)
    summary = []
    for i, (eng, envname, adapter, model, lang) in enumerate(jobs, 1):
        wav = HERE / 'audio' / f'{lang}_{args.variant}.wav'
        out = RESULTS / f'{eng}_{model}_{lang}_{args.variant}.json'
        tag = f'[{i}/{len(jobs)}] {eng}/{model}/{lang}/{args.variant}'
        if not args.force and already_ok(out):
            print(f'{tag}: SKIP (cached ok)', flush=True)
            summary.append((tag, 'cached'))
            continue
        t0 = time.perf_counter()
        print(f'{tag}: running ...', flush=True)
        proc = subprocess.run(
            [
                str(env_python(envname)),
                str(HERE / 'adapters' / adapter),
                str(wav),
                lang,
                model,
                str(out),
            ],
            capture_output=True,
            text=True,
        )
        dt = time.perf_counter() - t0
        line = (proc.stdout or proc.stderr).strip().splitlines()[-1:] or ['(no output)']
        print(f'{tag}: done in {dt:.0f}s -> {line[-1]}', flush=True)
        summary.append((tag, f'{dt:.0f}s'))

    print('\n=== run summary ===', flush=True)
    for tag, status in summary:
        print(f'  {tag}: {status}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
