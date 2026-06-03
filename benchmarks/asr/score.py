"""Score bake-off results against the reference subtitle tracks. Runs in `eval`.

EN-audio configs: WER + CER + chrF vs the English subtitle (loud caveat: the
Judas English sub is a *translation* of the Japanese, not a dub transcript, so
this bounds error rather than measuring it cleanly). JA-audio configs: timing
error vs the English subs (valid — both anchored to the same speech onsets) and
cross-engine CER agreement (consensus, not ground truth). All configs: RTF,
peak RAM, segment count.

    benchmarks/asr/envs/eval/bin/python benchmarks/asr/score.py --variant seg
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import eval as E

HERE = Path(__file__).parent
RESULTS = HERE / 'results'
REFS = HERE / 'refs'

# Segment window: prep_audio cut 180 s starting at 2:00.
SEG_START_MS = 120_000
SEG_END_MS = 300_000


def load_ref(path: Path, variant: str) -> list[dict]:
    segs = E.parse_ass_segments(path)
    if variant == 'full':
        return segs
    out = []
    for s in segs:
        if s['end_ms'] <= SEG_START_MS or s['start_ms'] >= SEG_END_MS:
            continue
        out.append(
            {
                'start_ms': max(0, s['start_ms'] - SEG_START_MS),
                'end_ms': s['end_ms'] - SEG_START_MS,
                'text': s['text'],
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=['seg', 'full'], default='seg')
    args = ap.parse_args()

    ref_en = load_ref(REFS / 'en_ref.ass', args.variant)
    ref_en_text = E.join_text(ref_en)

    rows = []
    ja_texts: dict[str, str] = {}
    for jf in sorted(RESULTS.glob(f'*_{args.variant}.json')):
        d = json.loads(jf.read_text())
        meta = d['meta']
        row = {
            'engine': d['engine'],
            'model': d['model'],
            'lang': d['lang'],
            'ok': meta['ok'],
            'rtf': meta.get('rtf'),
            'ram_mb': meta.get('peak_ram_mb'),
            'n_seg': len(d['segments']),
            'error': meta.get('error'),
        }
        if meta['ok'] and d['segments']:
            hyp_text = E.join_text(d['segments'])
            t = E.timing_errors(ref_en, d['segments'])
            row['timing_start_ms'] = t['mean_start_err_ms']
            row['timing_matched'] = t['matched']
            if d['lang'] == 'en':
                row['wer'] = round(E.wer(ref_en_text, hyp_text), 3)
                row['cer'] = round(E.cer(ref_en_text, hyp_text), 3)
                row['chrf'] = round(E.chrf([hyp_text], [ref_en_text]), 1)
            else:
                ja_texts[f'{d["engine"]}/{d["model"]}'] = hyp_text
        rows.append(row)

    # Cross-engine JA agreement: mean CER of each JA transcript vs all others.
    ja_agreement = {}
    for a, b in itertools.permutations(ja_texts, 2):
        ja_agreement.setdefault(a, []).append(E.cer(ja_texts[b], ja_texts[a]))
    ja_agreement = {k: round(sum(v) / len(v), 3) for k, v in ja_agreement.items()}

    out = {'variant': args.variant, 'rows': rows, 'ja_agreement_cer': ja_agreement}
    (RESULTS / f'scores_{args.variant}.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=1)
    )

    # Print a table.
    cols = [
        'engine',
        'model',
        'lang',
        'ok',
        'rtf',
        'ram_mb',
        'n_seg',
        'wer',
        'cer',
        'chrf',
        'timing_start_ms',
    ]
    print(' | '.join(c.ljust(13) for c in cols))
    for r in rows:
        print(' | '.join(str(r.get(c, '')).ljust(13) for c in cols))
    if ja_agreement:
        print('\nJA cross-engine agreement (mean CER vs others, lower=consensus):')
        for k, v in sorted(ja_agreement.items(), key=lambda kv: kv[1]):
            print(f'  {k}: {v}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
