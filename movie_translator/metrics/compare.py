"""Comparison logic for before/after metric reports."""

from __future__ import annotations

from typing import Any


def _top_level_stages(video: dict[str, Any]) -> set[str]:
    """Return the set of top-level stage names (no dots) from a video's entries."""
    stages: set[str] = set()
    for entry in video.get('entries', []):
        name: str = entry['name']
        top = name.split('.')[0]
        stages.add(top)
    return stages


def _identity_key(video: dict[str, Any]) -> tuple[Any, ...] | None:
    """Return a tuple key from identity fields, or None if identity is missing/empty."""
    identity = video.get('identity')
    if not identity:
        return None
    media_type = identity.get('media_type')
    title = identity.get('title')
    season = identity.get('season')
    episode = identity.get('episode')
    if not media_type or not title:
        return None
    return (media_type, title, season, episode)


def match_videos(
    before_videos: list[dict[str, Any]],
    after_videos: list[dict[str, Any]],
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], int]:
    """Match before/after videos by hash then by identity.

    Returns (matched_pairs, excluded_count) where excluded_count is the number
    of pairs that matched but were excluded due to different pipeline profiles.
    """
    # Build lookup maps for after videos
    after_by_hash: dict[str, dict[str, Any]] = {}
    after_by_identity: dict[tuple[Any, ...], dict[str, Any]] = {}
    for v in after_videos:
        h = v.get('hash')
        if h:
            after_by_hash[h] = v
        ik = _identity_key(v)
        if ik is not None:
            after_by_identity[ik] = v

    matched: list[tuple[dict[str, Any], dict[str, Any]]] = []
    excluded = 0
    used_after: set[int] = set()

    for before in before_videos:
        after: dict[str, Any] | None = None

        # Try hash match first
        h = before.get('hash')
        if h and h in after_by_hash:
            candidate = after_by_hash[h]
            if id(candidate) not in used_after:
                after = candidate

        # Fall back to identity match
        if after is None:
            ik = _identity_key(before)
            if ik is not None and ik in after_by_identity:
                candidate = after_by_identity[ik]
                if id(candidate) not in used_after:
                    after = candidate

        if after is None:
            continue

        # Check pipeline profile compatibility
        before_stages = _top_level_stages(before)
        after_stages = _top_level_stages(after)
        if before_stages != after_stages:
            excluded += 1
            continue

        used_after.add(id(after))
        matched.append((before, after))

    return matched, excluded


def compare_reports(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    """Compare two reports and return a structured comparison result.

    Returns a dict with:
      matched_videos, excluded_videos, total_before_videos, total_after_videos,
      spans (list of {name, before_ms, after_ms, delta_pct}),
      total_before_ms, total_after_ms
    """
    before_videos = before.get('videos', [])
    after_videos = after.get('videos', [])

    matched_pairs, excluded = match_videos(before_videos, after_videos)

    # Accumulate per-span totals across matched pairs
    span_before_totals: dict[str, float] = {}
    span_after_totals: dict[str, float] = {}
    span_counts: dict[str, int] = {}

    total_before_ms = 0.0
    total_after_ms = 0.0

    for bv, av in matched_pairs:
        total_before_ms += bv.get('total_duration_ms', 0)
        total_after_ms += av.get('total_duration_ms', 0)

        before_by_name = {e['name']: e['duration_ms'] for e in bv.get('entries', [])}
        after_by_name = {e['name']: e['duration_ms'] for e in av.get('entries', [])}

        # Only include spans present in both
        common_names = before_by_name.keys() & after_by_name.keys()
        for name in common_names:
            span_before_totals[name] = span_before_totals.get(name, 0.0) + before_by_name[name]
            span_after_totals[name] = span_after_totals.get(name, 0.0) + after_by_name[name]
            span_counts[name] = span_counts.get(name, 0) + 1

    # Build averaged spans list
    spans: list[dict[str, Any]] = []
    for name in span_before_totals:
        count = span_counts[name]
        b_avg = span_before_totals[name] / count
        a_avg = span_after_totals[name] / count
        delta_pct = ((a_avg - b_avg) / b_avg * 100) if b_avg else 0.0
        spans.append(
            {
                'name': name,
                'before_ms': b_avg,
                'after_ms': a_avg,
                'delta_pct': delta_pct,
            }
        )

    return {
        'matched_videos': len(matched_pairs),
        'excluded_videos': excluded,
        'total_before_videos': len(before_videos),
        'total_after_videos': len(after_videos),
        'spans': spans,
        'total_before_ms': total_before_ms,
        'total_after_ms': total_after_ms,
    }


def format_comparison(
    before: dict[str, Any],
    after: dict[str, Any],
    result: dict[str, Any],
) -> str:
    """Format a comparison result as a human-readable table string."""
    lines: list[str] = []

    # Header
    before_commit = before.get('git_commit', 'unknown')
    after_commit = after.get('git_commit', 'unknown')
    before_date = (before.get('timestamp', '') or '')[:10]
    after_date = (after.get('timestamp', '') or '')[:10]
    lines.append(f'Comparing: {before_commit} ({before_date}) -> {after_commit} ({after_date})')

    # Config line — merge both configs and show key=value pairs
    config: dict[str, Any] = {**before.get('config', {}), **after.get('config', {})}
    if config:
        config_str = ', '.join(f'{k}={v}' for k, v in sorted(config.items()))
        lines.append(f'Config: {config_str}')

    # Dirty warnings
    before_dirty = 'yes (results may be unreliable)' if before.get('dirty') else 'no'
    after_dirty = 'yes (results may be unreliable)' if after.get('dirty') else 'no'
    lines.append(f'Dirty: before={before_dirty}, after={after_dirty}')

    lines.append('')

    # Match summary
    matched = result['matched_videos']
    excluded = result['excluded_videos']
    total_before = result['total_before_videos']
    lines.append(
        f'Videos matched: {matched}/{total_before} ({excluded} excluded for different profiles)'
    )
    lines.append('')

    # Table
    col_stage = 18
    col_num = 12

    header = (
        f'{"Stage":<{col_stage}} {"Before":>{col_num}} {"After":>{col_num}} {"Delta":>{col_num}}'
    )
    separator = '-' * len(header)
    lines.append(header)
    lines.append(separator)

    spans = result.get('spans', [])
    for span in spans:
        name = span['name']
        b_ms = span['before_ms']
        a_ms = span['after_ms']
        delta_pct = span['delta_pct']
        sign = '+' if delta_pct >= 0 else ''
        row = (
            f'{name:<{col_stage}} '
            f'{b_ms:>{col_num - 2}.0f}ms '
            f'{a_ms:>{col_num - 2}.0f}ms '
            f'{sign}{delta_pct:>{col_num - 2}.0f}%'
        )
        lines.append(row)

    lines.append(separator)

    # Total row
    total_b = result['total_before_ms']
    total_a = result['total_after_ms']
    total_delta_pct = ((total_a - total_b) / total_b * 100) if total_b else 0.0
    sign = '+' if total_delta_pct >= 0 else ''
    total_label = 'Total (wall clock)'
    total_row = (
        f'{total_label:<{col_stage}} '
        f'{total_b:>{col_num - 2}.0f}ms '
        f'{total_a:>{col_num - 2}.0f}ms '
        f'{sign}{total_delta_pct:>{col_num - 2}.0f}%'
    )
    lines.append(total_row)
    lines.append('')

    # Summary line
    overall_pct = abs(total_delta_pct)
    if total_delta_pct < 0:
        direction = 'faster'
    elif total_delta_pct > 0:
        direction = 'slower'
    else:
        direction = 'unchanged'

    summary_parts = [f'{overall_pct:.0f}% {direction} overall.']

    # Biggest win (most negative delta_pct)
    if spans:
        biggest_win = min(spans, key=lambda s: s['delta_pct'])
        if biggest_win['delta_pct'] < 0:
            bw_name = biggest_win['name']
            bw_pct = abs(biggest_win['delta_pct'])
            bw_delta_ms = biggest_win['after_ms'] - biggest_win['before_ms']
            summary_parts.append(f'Biggest win: {bw_name} ({bw_pct:.0f}%, {bw_delta_ms:+.0f}ms).')

    lines.append('Summary: ' + ' '.join(summary_parts))
    lines.append(f'{matched} videos compared. {excluded} excluded.')

    return '\n'.join(lines)
