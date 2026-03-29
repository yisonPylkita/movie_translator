"""Structural classification of ASS subtitle styles.

Classifies styles as dialogue or non-dialogue based on aggregate event
properties rather than style name keywords. This is robust to arbitrary
naming conventions across different fansub groups.

The classifier uses three signals:
  1. Positioning ratio — signs and karaoke use explicit \\pos()/\\move()
  2. Average text length — karaoke syllables are 1-3 characters
  3. Event count and duration — karaoke has many rapid-fire events

A rescue check prevents filtering out legitimate dialogue that uses
alternate positioning (e.g., top-positioned lines with \\an8).
"""

from __future__ import annotations


def classify_styles(subs) -> set[str]:
    """Classify ASS styles and return the set of dialogue style names.

    Analyzes aggregate properties of events per style to determine which
    styles contain dialogue. Works with any pysubs2 SSAFile object.

    Args:
        subs: A pysubs2.SSAFile object.

    Returns:
        Set of style names classified as dialogue.
    """
    # Collect per-style metrics
    style_metrics: dict[str, dict] = {}

    for event in subs:
        text = event.text or ''
        if not text.strip():
            continue

        style = getattr(event, 'style', 'Default')
        if style not in style_metrics:
            style_metrics[style] = {
                'count': 0,
                'total_duration': 0,
                'total_text_len': 0,
                'positioned_count': 0,
            }

        m = style_metrics[style]
        m['count'] += 1
        m['total_duration'] += event.end - event.start

        plaintext = event.plaintext.strip() if hasattr(event, 'plaintext') else text
        m['total_text_len'] += len(plaintext)

        if '\\pos(' in text or '\\move(' in text:
            m['positioned_count'] += 1

    dialogue_styles: set[str] = set()

    for style, m in style_metrics.items():
        n = m['count']
        if n == 0:
            continue

        avg_text = m['total_text_len'] / n
        avg_dur = m['total_duration'] / n
        pos_ratio = m['positioned_count'] / n

        if _is_dialogue(pos_ratio, avg_text, avg_dur, n):
            dialogue_styles.add(style)

    return dialogue_styles


def _is_dialogue(
    pos_ratio: float,
    avg_text: float,
    avg_dur: float,
    count: int,
) -> bool:
    """Determine if a style is dialogue based on its aggregate metrics."""
    # Rule 1: High positioning = non-dialogue (signs/typesetting)
    if pos_ratio >= 0.5:
        # Rescue: long text + long duration = positioned dialogue (e.g., \an8 top lines)
        if avg_text > 20 and avg_dur > 1500:
            return True
        return False

    # Rule 2: Very short text + many events = karaoke syllables
    if avg_text < 5 and count > 50:
        return False

    # Rule 3: Rapid-fire short events = karaoke
    if count > 500 and avg_dur < 500:
        return False

    return True
