"""Thin CLI wrapper around movie_translator.translation.translate_dialogue_lines.

Mirrors the `TranslateTask` GPU task (movie_translator/gpu_queue.py).

Contract
--------
Reads a single JSON object from stdin::

    {
        "lines":         [{"start_ms": int, "end_ms": int, "text": str}, ...],
        "device":        str,            # "cpu" | "mps" | "cuda"
        "batch_size":    int,
        "model":         str,            # "allegro" | "apple" | "nllb" | ...
        "proper_nouns":  [str, ...] | null
    }

Writes a single JSON object to stdout::

    {"lines": [{"start_ms": int, "end_ms": int, "text": str}, ...]}

Exit codes:
    0  success
    1  bad input JSON
    2  unexpected runtime error

`--self-test`
    Skip importing torch / the translation backend entirely. Each input
    line's text is prefixed with ``[xl] `` so the Rust runner + JSON
    contract can be integration-tested without any models present.

Invoked by the Rust `mt-ml` crate via ``uv run python ml/translate.py``
(from the repo root).
"""

from __future__ import annotations

import json
import sys


def _read_request() -> dict:
    return json.loads(sys.stdin.read())


def _self_test(req: dict) -> dict:
    lines = req.get("lines", [])
    return {
        "lines": [
            {
                "start_ms": int(line["start_ms"]),
                "end_ms": int(line["end_ms"]),
                "text": f"[xl] {line['text']}",
            }
            for line in lines
        ]
    }


def _real(req: dict) -> dict:
    # Import here so import errors surface as exit-code 2, not at parse time.
    from movie_translator.translation import translate_dialogue_lines  # noqa: PLC0415
    from movie_translator.types import DialogueLine  # noqa: PLC0415

    lines = [
        DialogueLine(int(line["start_ms"]), int(line["end_ms"]), line["text"])
        for line in req.get("lines", [])
    ]

    raw_proper_nouns = req.get("proper_nouns")
    proper_nouns = set(raw_proper_nouns) if raw_proper_nouns else None

    translated = translate_dialogue_lines(
        dialogue_lines=lines,
        device=req["device"],
        batch_size=int(req["batch_size"]),
        model=req["model"],
        proper_nouns=proper_nouns,
    )

    return {
        "lines": [
            {"start_ms": line.start_ms, "end_ms": line.end_ms, "text": line.text}
            for line in translated
        ]
    }


def main(argv: list[str]) -> int:
    self_test = "--self-test" in argv

    try:
        req = _read_request()
    except json.JSONDecodeError as exc:
        print(f"error: invalid JSON on stdin: {exc}", file=sys.stderr)
        return 1

    try:
        output = _self_test(req) if self_test else _real(req)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(output))
    return 0


if __name__ == "__main__":
    # Repo-root path insertion so `import movie_translator` works regardless
    # of the working directory the Rust runner spawns us from.
    import pathlib

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    sys.exit(main(sys.argv[1:]))
