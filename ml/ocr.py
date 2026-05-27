"""Thin CLI wrapper around movie_translator OCR extraction.

Mirrors the `OcrTask` GPU task (movie_translator/gpu_queue.py), which has
two modes selected by ``ocr_type``: ``pgs`` and ``burned_in``.

Contract
--------
PGS mode::

    python ml/ocr.py --type pgs --video PATH --track-index N --work-dir DIR

    stdout: {"srt_path": str | null}

    (extract_pgs_track returns a Path or None when extraction failed.)

Burned-in mode::

    python ml/ocr.py --type burned_in --video PATH --output-dir DIR \
        --crop-ratio F --fps N

    stdout: {
        "srt_path": str,
        "ocr_results": [
            {
                "timestamp_ms": int,
                "text": str,
                "boxes": [{"x": f, "y": f, "width": f, "height": f}, ...]
            },
            ...
        ]
    }

    (extract_burned_in_subtitles returns a BurnedInResult or None; on None
     this script exits with code 2 and an error on stderr.)

Exit codes:
    0  success
    1  bad arguments
    2  unexpected runtime error (incl. extraction returning None for burned_in)

`--self-test`
    Emit a fixed fake result without importing cv2 / the vision backend, so
    the Rust runner + JSON contract can be integration-tested without models.

Invoked by the Rust `mt-ml` crate via ``uv run python ml/ocr.py`` (from the
repo root).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OCR a subtitle track (PGS or burned-in).")
    parser.add_argument("--type", dest="ocr_type", choices=["pgs", "burned_in"], required=True)
    parser.add_argument("--video", required=True)
    # PGS-specific
    parser.add_argument("--track-index", type=int, default=0)
    parser.add_argument("--work-dir", default="")
    # Burned-in-specific
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--crop-ratio", type=float, default=0.25)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--self-test", action="store_true")
    return parser


def _self_test(args: argparse.Namespace) -> dict:
    if args.ocr_type == "pgs":
        return {"srt_path": f"{args.video}.srt"}
    return {
        "srt_path": f"{args.video}.srt",
        "ocr_results": [
            {
                "timestamp_ms": 1000,
                "text": "self-test subtitle",
                "boxes": [{"x": 0.1, "y": 0.8, "width": 0.5, "height": 0.1}],
            }
        ],
    }


def _real_pgs(args: argparse.Namespace) -> dict:
    from movie_translator.ocr.pgs_extractor import extract_pgs_track  # noqa: PLC0415

    srt_path = extract_pgs_track(
        video_path=Path(args.video),
        track_index=args.track_index,
        work_dir=Path(args.work_dir),
    )
    return {"srt_path": str(srt_path) if srt_path is not None else None}


def _real_burned_in(args: argparse.Namespace) -> dict:
    from movie_translator.ocr import extract_burned_in_subtitles  # noqa: PLC0415

    result = extract_burned_in_subtitles(
        video_path=Path(args.video),
        output_dir=Path(args.output_dir),
        crop_ratio=args.crop_ratio,
        fps=args.fps,
    )
    if result is None:
        raise RuntimeError("burned-in extraction returned no result")

    return {
        "srt_path": str(result.srt_path),
        "ocr_results": [
            {
                "timestamp_ms": r.timestamp_ms,
                "text": r.text,
                "boxes": [
                    {"x": b.x, "y": b.y, "width": b.width, "height": b.height}
                    for b in r.boxes
                ],
            }
            for r in result.ocr_results
        ],
    }


def main(argv: list[str]) -> int:
    try:
        args = _build_parser().parse_args(argv)
    except SystemExit:
        return 1

    try:
        if args.self_test:
            output = _self_test(args)
        elif args.ocr_type == "pgs":
            output = _real_pgs(args)
        else:
            output = _real_burned_in(args)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(output))
    return 0


if __name__ == "__main__":
    # Repo-root path insertion so `import movie_translator` works regardless
    # of the working directory the Rust runner spawns us from.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.exit(main(sys.argv[1:]))
