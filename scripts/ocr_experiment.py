"""OCR-only ablation harness for the burned-in Polish hardsub OCR.

Runs the OCR stage (frame extract -> change detection -> Vision OCR -> clean)
on ONE downloaded ogladajanime clip under several configs and scores each
against a reference line set, so we can see which change helps and tune values
WITHOUT touching the full pipeline.

Vision OCR is the expensive part, so we extract once at the highest fps and
cache OCR results per frame; every config is then assembled from the cache.

Usage:
    .venv/bin/python scripts/ocr_experiment.py <clip.mp4>
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from PIL import Image

from movie_translator.hardsub.postprocess import is_dialogue, merge_ocr_results
from movie_translator.ocr.frame_extractor import extract_subtitle_region_frames
from movie_translator.ocr.vision_ocr import recognize_text_with_boxes

# Reference: the burned-in Polish dialogue actually shown in the test window
# (transcribed from the source frames), normalized. Coverage = how many of
# these a config recovers.
REFERENCE = [
    'opowiadanie historii to jedyna rzecz w której czuję się pewna siebie',
    'przepraszam nie płacz',
    'nie dostałem takiego bonusu jak miecz boga lodu',
    'musiałeś dostać moc magiczną jako boski dodatek',
    'nie dostałem żadnego bonusu nawet nie rozmawiałem z bogiem',
    'więc zobaczmy na moment kiedy zostałeś przywany 18 lat temu',
    'masz rację',
    '18 lat temu',
    'to było dawno temu i może zająć chwilę do znalezienia',
    'grindowanie mobów egzekucja kolejna egzekucja',
    'bełkocze niepokojące rzeczy',
    'polowali na niego od początku jako orka',
    'znalazłem',
    'ty gówniany orku próbujesz zjeść moją krowę',
    'nie wujku to nie to',
]

SCALE_WIDTH = 1280
HI_FPS = 6  # superset extraction fps


def norm(s: str) -> str:
    return ' '.join(s.lower().replace('\n', ' ').split())


def covered(ref: str, outputs: list[str], thresh: float = 0.55) -> bool:
    return any(SequenceMatcher(None, ref, norm(o)).ratio() >= thresh for o in outputs)


def is_sign_line(text: str) -> bool:
    """A single all-caps token (no spaces) like 'CALENDAR' — an on-screen sign,
    not dialogue. Real Polish dialogue is multi-word/lowercase/diacritic'd."""
    t = text.strip()
    if not t or ' ' in t:
        return False
    letters = [c for c in t if c.isalpha()]
    if len(letters) < 3:
        return False
    return sum(c.isupper() for c in letters) / len(letters) >= 0.8


def is_garbage(text: str) -> bool:
    """Sign-like or non-dialogue output line (the kind we want gone). Does NOT
    penalize valid dialogue outside the reference window."""
    for part in text.split('\n'):
        if is_sign_line(part):
            return True
    return not is_dialogue(text)


def load_gray(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert('L'))


def transitions(frames: list[tuple[Path, int]], threshold: float) -> list[tuple[Path, int]]:
    """Frames where the region changed by > threshold (parametrized copy)."""
    if len(frames) < 2:
        return list(frames)
    out: list[tuple[Path, int]] = []
    prev = load_gray(frames[0][0])
    if float(np.var(prev)) > 200.0:
        out.append(frames[0])
    for path, ts in frames[1:]:
        curr = load_gray(path)
        diff = float(np.mean(np.abs(curr.astype(np.int16) - prev.astype(np.int16))))
        if diff > threshold and float(np.var(curr)) > 200.0:
            out.append((path, ts))
        prev = curr
    return out


@dataclass
class Config:
    name: str
    fps: int
    threshold: float
    sign_filter: bool  # drop all-caps single-token sign boxes (CALENDAR)
    min_duration_ms: int


def assemble(
    cfg: Config,
    frames_hi: list[tuple[Path, int]],
    ocr_cache: dict[str, list[tuple[str, float]]],
) -> list[str]:
    """Build cleaned lines for a config from cached per-box OCR (text, center_x)."""
    step = max(1, HI_FPS // cfg.fps)
    frames = frames_hi[::step]
    trans = transitions(frames, cfg.threshold)
    frame_texts: list[tuple[int, str]] = []
    for path, ts in trans:
        boxes = ocr_cache.get(str(path), [])
        kept = [t for (t, _cx) in boxes if not (cfg.sign_filter and is_sign_line(t))]
        frame_texts.append((ts, '\n'.join(kept).strip()))
    lines = merge_ocr_results(frame_texts, min_duration_ms=cfg.min_duration_ms)
    return [ln.text for ln in lines]


def main() -> None:
    clip = Path(sys.argv[1]).expanduser()
    work = Path(tempfile.mkdtemp(prefix='ocr_exp_'))
    print(f'clip: {clip}\nwork: {work}\n')

    frames_hi = extract_subtitle_region_frames(
        clip, work / 'frames', fps=HI_FPS, crop_ratio=0.25, scale_width=SCALE_WIDTH
    )
    print(f'extracted {len(frames_hi)} frames @ {HI_FPS}fps')

    configs = [
        # name, fps, change-threshold, sign-filter, min_duration
        Config('C0 baseline (current)', 3, 15.0, False, 500),
        Config('C1 +sign-filter', 3, 15.0, True, 500),
        Config('C2 +fps6 +thresh8', 6, 8.0, True, 500),
        Config('C3 +min-dur 250', 6, 8.0, True, 250),
        Config('C4 +thresh4 +min-dur150', 6, 4.0, True, 150),
        Config('C5 +thresh2 +min-dur100', 6, 2.0, True, 100),
    ]

    # OCR every frame that any config will examine (union of transition sets),
    # caching (text, center_x) per box. center_x is full-frame (crop is full
    # width, so box.x is unchanged by the crop).
    needed: set[str] = set()
    for cfg in configs:
        step = max(1, HI_FPS // cfg.fps)
        for path, _ in transitions(frames_hi[::step], cfg.threshold):
            needed.add(str(path))
    print(f'OCR-ing {len(needed)} unique transition frames (cached)…\n')
    ocr_cache: dict[str, list[tuple[str, float]]] = {}
    for i, path in enumerate(sorted(needed)):
        boxes = recognize_text_with_boxes(Path(path), language='pl')
        ocr_cache[path] = [(t, b.x + b.width / 2.0) for (t, b) in boxes]
        if (i + 1) % 50 == 0:
            print(f'  …{i + 1}/{len(needed)}')

    print('\n================ RESULTS ================\n')
    for cfg in configs:
        outs = assemble(cfg, frames_hi, ocr_cache)
        cov = sum(covered(r, outs) for r in REFERENCE)
        garbage = [o for o in outs if is_garbage(o)]
        print(f'### {cfg.name}')
        print(f'   lines={len(outs)}  coverage={cov}/{len(REFERENCE)}  garbage={len(garbage)}')
        if garbage:
            print(f'   garbage: {[g[:40] for g in garbage]}')
        print()

    # Detailed dump of the tuned config for eyeballing.
    print('================ C4 tuned — full output ================')
    for o in assemble(configs[-1], frames_hi, ocr_cache):
        print(f'  • {o!r}')


if __name__ == '__main__':
    main()
