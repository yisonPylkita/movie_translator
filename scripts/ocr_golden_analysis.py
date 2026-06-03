"""Golden-sample OCR analysis: capture EVERY burned-in line, then explain each
miss in the production config.

Phase 1 (golden): extract frames at a high fps and OCR *every* frame (no
change-detection gating) — this is the ground truth of what was on screen.
Phase 2 (production): simulate the shipped config (fps6, change-threshold 4,
sign-filter, min-dur 200) and find which golden lines it loses.
Phase 3 (diagnosis): for each missing line, classify WHY by tracing it through
the production stages — was a transition even detected? was it OCR'd but too
short? merged into a neighbor? an OCR miss?

Usage: .venv/bin/python scripts/ocr_golden_analysis.py <clip.mp4> [golden_fps]
"""

from __future__ import annotations

import sys
import tempfile
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from PIL import Image

from movie_translator.hardsub.postprocess import merge_ocr_results
from movie_translator.ocr.burned_in_extractor import _is_sign_text
from movie_translator.ocr.frame_extractor import extract_subtitle_region_frames
from movie_translator.ocr.vision_ocr import recognize_text_with_boxes

# Production config (the shipped defaults).
PROD_FPS = 6
PROD_THRESHOLD = 4.0
PROD_MIN_DUR = 200
SCALE_WIDTH = 1280


def norm(s: str) -> str:
    return ' '.join(s.lower().replace('\n', ' ').split())


def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def load_gray(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert('L'))


def main() -> None:
    clip = Path(sys.argv[1]).expanduser()
    golden_fps = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    work = Path(tempfile.mkdtemp(prefix='ocr_golden_'))
    print(f'clip={clip}\ngolden_fps={golden_fps}\n')

    frames = extract_subtitle_region_frames(
        clip, work / 'f', fps=golden_fps, crop_ratio=0.25, scale_width=SCALE_WIDTH
    )
    print(f'extracted {len(frames)} frames @ {golden_fps}fps')

    # ── Phase 1: OCR EVERY frame (golden). Cache gray + sign-filtered text. ──
    gray: list[np.ndarray] = []
    texts: list[str] = []
    print('OCR-ing every frame (golden)…')
    for i, (path, _ts) in enumerate(frames):
        g = load_gray(path)
        gray.append(g)
        if float(np.var(g)) <= 200.0:  # no text in region
            texts.append('')
        else:
            boxes = recognize_text_with_boxes(path, language='pl')
            kept = [t for (t, _b) in boxes if not _is_sign_text(t)]
            texts.append('\n'.join(kept).strip())
        if (i + 1) % 200 == 0:
            print(f'  …{i + 1}/{len(frames)}')

    ts = [t for _p, t in frames]
    golden_ft = list(zip(ts, texts, strict=True))
    golden = merge_ocr_results(golden_ft, min_duration_ms=0, similarity=0.80)
    golden = [g for g in golden if g.text.strip()]
    print(f'\nGOLDEN: {len(golden)} distinct lines on screen\n')

    # ── Phase 2/3: sweep change metrics, find one that catches all lines ────
    step = max(1, golden_fps // PROD_FPS)
    idx6 = list(range(0, len(frames), step))

    # Per-6fps-frame change metrics vs the previous 6fps frame:
    #   meandiff = mean |Δ| over the crop (current production metric)
    #   fracdiff = fraction of pixels whose |Δ| > 25 (text appearing lights up
    #              many pixels even when the *mean* barely moves)
    meandiff: dict[int, float] = {}
    fracdiff: dict[int, float] = {}
    prevk = idx6[0]
    for k in idx6[1:]:
        d = np.abs(gray[k].astype(np.int16) - gray[prevk].astype(np.int16))
        meandiff[k] = float(d.mean())
        fracdiff[k] = float((d > 25).mean())
        prevk = k

    def trans_by(metric: dict[int, float], thr: float) -> list[int]:
        out = [idx6[0]] if float(np.var(gray[idx6[0]])) > 200.0 else []
        for k in idx6[1:]:
            if metric[k] > thr and float(np.var(gray[k])) > 200.0:
                out.append(k)
        return out

    def simulate(trans: list[int]):  # noqa: ANN202
        prod_ft = [(ts[k], texts[k]) for k in trans]
        prod = [
            p for p in merge_ocr_results(prod_ft, min_duration_ms=PROD_MIN_DUR) if p.text.strip()
        ]
        missing = [
            g
            for g in golden
            if not any(
                sim(g.text, p.text) >= 0.6 and not (p.end_ms < g.start_ms or p.start_ms > g.end_ms)
                for p in prod
            )
        ]
        return prod, missing

    print('================ METRIC SWEEP (lines / missing / transitions) ===')
    print(f'{"metric@thr":<22}{"prod":>6}{"missing":>9}{"transitions":>13}')
    candidates = [('meandiff', t) for t in (4.0, 2.0, 1.0)] + [
        ('fracdiff', t) for t in (0.010, 0.006, 0.004, 0.003, 0.002, 0.0015)
    ]
    results = {}
    for name, thr in candidates:
        metric = meandiff if name == 'meandiff' else fracdiff
        trans = trans_by(metric, thr)
        prod, missing = simulate(trans)
        results[(name, thr)] = (prod, missing, trans)
        tag = f'{name}@{thr}'
        print(f'{tag:<22}{len(prod):>6}{len(missing):>9}{len(trans):>13}')

    # Detail the missing lines for the current prod metric and the best frac one.
    for label, key in (('CURRENT meandiff@4.0', ('meandiff', 4.0)),):
        _, missing, _ = results[key]
        print(f'\n--- {label}: {len(missing)} missing ---')
        for g in missing:
            print(f'  [{g.start_ms // 1000:>4}s {g.end_ms - g.start_ms}ms] {g.text[:55]!r}')

    best = min(results.items(), key=lambda kv: (len(kv[1][1]), len(kv[1][2])))
    (bname, bthr), (bprod, bmissing, btrans) = best
    print(f'\n--- BEST: {bname}@{bthr}  missing={len(bmissing)}  transitions={len(btrans)} ---')
    for g in bmissing:
        print(f'  STILL MISSING [{g.start_ms // 1000}s {g.end_ms - g.start_ms}ms] {g.text[:55]!r}')


if __name__ == '__main__':
    main()
