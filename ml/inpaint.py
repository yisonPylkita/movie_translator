"""Thin CLI wrapper around movie_translator.inpainting.remove_burned_in_subtitles.

Mirrors the `InpaintTask` GPU task (movie_translator/gpu_queue.py).

Contract
--------
::

    python ml/inpaint.py --video PATH --output PATH --device STR \
        --backend STR --ocr-results JSON_PATH

The ``--ocr-results`` file contains a JSON list of OCR results::

    [
        {
            "timestamp_ms": int,
            "text": str,
            "boxes": [{"x": f, "y": f, "width": f, "height": f}, ...]
        },
        ...
    ]

On success writes ``{"output_path": str}`` to stdout and exits 0.

Exit codes:
    0  success
    1  bad arguments / bad ocr-results JSON
    2  unexpected runtime error

`--self-test`
    Copy the input video to the output path without importing torch / the
    inpainting backend, so the Rust runner + JSON contract can be
    integration-tested without models.

Invoked by the Rust `mt-ml` crate via ``uv run python ml/inpaint.py`` (from
the repo root).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remove burned-in subtitles via inpainting.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", default="lama")
    parser.add_argument("--ocr-results", dest="ocr_results", required=True)
    parser.add_argument("--self-test", action="store_true")
    return parser


def _load_ocr_results(path: str) -> list:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _real(args: argparse.Namespace, raw_results: list) -> None:
    from movie_translator.inpainting import remove_burned_in_subtitles  # noqa: PLC0415
    from movie_translator.types import BoundingBox, OCRResult  # noqa: PLC0415

    ocr_results = [
        OCRResult(
            timestamp_ms=int(r["timestamp_ms"]),
            text=r["text"],
            boxes=[
                BoundingBox(b["x"], b["y"], b["width"], b["height"]) for b in r.get("boxes", [])
            ],
        )
        for r in raw_results
    ]

    remove_burned_in_subtitles(
        video_path=Path(args.video),
        output_path=Path(args.output),
        ocr_results=ocr_results,
        device=args.device,
        backend=args.backend,
    )


def main(argv: list[str]) -> int:
    try:
        args = _build_parser().parse_args(argv)
    except SystemExit:
        return 1

    try:
        raw_results = _load_ocr_results(args.ocr_results)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: failed to read ocr-results: {exc}", file=sys.stderr)
        return 1

    try:
        if args.self_test:
            # Copy input to output without importing torch.
            shutil.copy2(args.video, args.output)
        else:
            _real(args, raw_results)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps({"output_path": args.output}))
    return 0


if __name__ == "__main__":
    # Repo-root path insertion so `import movie_translator` works regardless
    # of the working directory the Rust runner spawns us from.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.exit(main(sys.argv[1:]))
