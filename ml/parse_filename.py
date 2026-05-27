"""Thin CLI wrapper around movie_translator.identifier.parser.parse_filename.

Contract
--------
Reads a single JSON object from stdin::

    {"filename": "...", "folder_name": null | "..."}

Writes a single JSON object to stdout::

    {
        "title":         str | null,
        "year":          int | null,
        "season":        int | null,
        "episode":       int | null,
        "media_type":    "movie" | "episode",
        "is_anime":      bool,
        "release_group": str | null
    }

Exit codes:
    0  success
    1  bad input JSON
    2  unexpected runtime error

Invoked by the Rust `mt-discovery` crate via
``uv run python ml/parse_filename.py`` (from the repo root).
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    raw = sys.stdin.read()
    try:
        req = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"error: invalid JSON on stdin: {exc}", file=sys.stderr)
        return 1

    filename: str = req.get("filename", "")
    folder_name: str | None = req.get("folder_name")  # None is fine

    try:
        # Import here so errors surface as exit-code 2, not import-time
        from movie_translator.identifier.parser import parse_filename  # noqa: PLC0415

        result = parse_filename(filename, folder_name=folder_name)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Normalise guessit int-like types to plain Python ints / None
    def _int_or_none(v: object) -> int | None:
        return int(v) if v is not None else None

    output = {
        "title": result.get("title"),
        "year": _int_or_none(result.get("year")),
        "season": _int_or_none(result.get("season")),
        "episode": _int_or_none(result.get("episode")),
        "media_type": result.get("media_type", "movie"),
        "is_anime": bool(result.get("is_anime", False)),
        "release_group": result.get("release_group"),
    }

    print(json.dumps(output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
