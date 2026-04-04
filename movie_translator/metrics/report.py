"""Report serialization and git metadata helpers."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _git_short_hash() -> str:
    """Return the short git commit hash of HEAD."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return 'unknown'


def _git_is_dirty() -> bool:
    """Return True if the working tree has uncommitted changes."""
    try:
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return bool(result.stdout.strip())
    except Exception:
        return True


def build_report(
    *,
    videos: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build a complete report dict ready for serialization."""
    return {
        'version': 1,
        'git_commit': _git_short_hash(),
        'dirty': _git_is_dirty(),
        'timestamp': datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'config': config,
        'videos': videos,
    }


def save_report(report: dict[str, Any], path: Path) -> None:
    """Write a report dict to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')


def load_report(path: Path) -> dict[str, Any]:
    """Load a report dict from a JSON file."""
    return json.loads(path.read_text())
