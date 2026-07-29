# Known Issues

> Owner: parent orchestrator. Update when issues confirmed, resolved, or workarounds found.

## Leading signs/karaoke skew subtitle validation

**Status:** confirmed

**Symptoms:** Timing-overlap validation score drops below 0.8 when first dialogue line
is preceded by signs or karaoke effects.

**Reproduction:** Subtitles where opening contains non-dialogue timed events before actual dialogue.

**Evidence:** Observed on anime with opening karaoke (Konosuba S1E1).

**Workaround:** Manual track selection when automated validation rejects viable tracks.

**Next action:** Investigate skipping pre-dialogue events in scoring window.

## Static subtitle offsets (1-3s, 60-90s)

**Status:** confirmed

**Symptoms:** Fetched Polish subtitles have constant offset from source. Common on anime.

**Reproduction:** Konosuba S1E1 — constant Polish-sub offset.

**Evidence:** ilass DP alignment handles most cases; cross-correlation fallback catches remaining.

**Workaround:** Alignment pipeline automatically corrects. No user action needed.

**Next action:** Monitor alignment failure rate on new content types.
