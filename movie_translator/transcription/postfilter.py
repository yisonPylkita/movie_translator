"""Clean raw Whisper output for subtitle use.

Whisper hallucinates on trailing music/silence (the bake-off caught it looping
ご視聴ありがとうございました over the ED with timestamps past the end of the audio).
This filter drops empties, drops segments that start at/past the audio end,
clamps end times, and collapses consecutive duplicate texts. Collapsing
consecutive duplicates can eat a genuine immediate repeat, but in practice
consecutive identical Whisper segments are overwhelmingly hallucination.
"""

from __future__ import annotations

from ..types import DialogueLine


def clean_segments(segs: list[DialogueLine], audio_ms: int) -> list[DialogueLine]:
    """Drop/clamp/de-loop raw ASR segments against the real audio length."""
    out: list[DialogueLine] = []
    prev_text: str | None = None
    for seg in segs:
        text = seg.text.strip()
        if not text:
            continue
        if seg.start_ms >= audio_ms:
            continue
        if text == prev_text:
            continue
        out.append(DialogueLine(seg.start_ms, min(seg.end_ms, audio_ms), text))
        prev_text = text
    return out
