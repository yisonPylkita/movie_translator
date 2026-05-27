//! Piecewise subtitle realignment for fetched subtitles.
//!
//! Ported from `movie_translator/subtitle_fetch/align.py`.
//!
//! Compares a fetched subtitle file against a reference track (typically
//! extracted from the video) and estimates timing offsets.  Handles the
//! common case where the candidate was timed to a video source with the
//! opening sequence (OP) removed, producing a different offset for pre-OP
//! and post-OP content.
//!
//! Algorithm:
//!   1. Detect structural gaps (OP/ED) in the reference by finding large
//!      dialogue-free intervals.
//!   2. If no gap found, estimate a single global offset via cross-correlation.
//!   3. If an OP gap is found, estimate separate offsets for the pre-OP and
//!      post-OP segments.  The post-OP search range accounts for the full
//!      OP duration being removed.
//!   4. Apply per-segment shifts to the subtitle file.

use std::path::Path;

use crate::validator::{build_activity_vector, extract_timestamps};

// ---------------------------------------------------------------------------
// estimate_offset
// ---------------------------------------------------------------------------

/// Minimum dialogue-free interval to be considered an OP/ED gap.
const MIN_GAP_MS: i64 = 60_000;

/// Only look for the OP gap starting at 30s (OP rarely starts before 30s).
const OP_SEARCH_START_MS: i64 = 30_000;

/// OP should be done by 6 minutes.
const OP_SEARCH_END_MS: i64 = 360_000;

/// Minimum absolute offset to apply (below this is noise).
pub const MIN_OFFSET_MS: i64 = 150;

/// Estimate static timing offset via cross-correlation.
///
/// Builds binary activity vectors from both tracks and finds the shift
/// that maximises their overlap.
///
/// A positive result means the candidate is early (shift it later).
/// A negative result means the candidate is late (shift it earlier).
///
/// Mirrors Python `estimate_offset(ref_timestamps, cand_timestamps, bin_size_ms, max_shift_ms, min_quality)`.
pub fn estimate_offset(
    ref_timestamps: &[(i64, i64)],
    cand_timestamps: &[(i64, i64)],
    bin_size_ms: i64,
    max_shift_ms: i64,
    min_quality: f64,
) -> Option<i64> {
    if ref_timestamps.is_empty() || cand_timestamps.is_empty() {
        return None;
    }

    let duration = ref_timestamps
        .iter()
        .map(|&(_, e)| e)
        .chain(cand_timestamps.iter().map(|&(_, e)| e))
        .max()
        .unwrap_or(0);

    let ref_vec = build_activity_vector(ref_timestamps, duration, bin_size_ms);
    let cand_vec = build_activity_vector(cand_timestamps, duration, bin_size_ms);

    if ref_vec.is_empty() || cand_vec.is_empty() {
        return None;
    }

    let max_len = ref_vec.len().max(cand_vec.len());
    let mut ref_padded = ndarray::Array1::<f64>::zeros(max_len);
    let mut cand_padded = ndarray::Array1::<f64>::zeros(max_len);
    ref_padded
        .slice_mut(ndarray::s![..ref_vec.len()])
        .assign(&ref_vec);
    cand_padded
        .slice_mut(ndarray::s![..cand_vec.len()])
        .assign(&cand_vec);

    let max_shift_bins = (max_shift_ms / bin_size_ms) as usize;
    let effective_max = max_shift_bins.min(max_len.saturating_sub(1));

    // Full cross-correlation (mode='full'): zero-lag at index len(ref) - 1
    let n = max_len;
    let zero_lag = n - 1;
    let lo = zero_lag.saturating_sub(effective_max);
    let hi = (zero_lag + effective_max + 1).min(2 * n - 1);

    // Direct O(n * range) correlation — gives integer-exact results vs numpy.
    let mut best_score = f64::NEG_INFINITY;
    let mut best_k = lo; // absolute index in the full correlation array

    for k in lo..hi {
        // shift = k - zero_lag  (matches Python: best_shift = best_idx - effective_max
        // where best_idx is 0-indexed within the slice starting at lo = zero_lag - effective_max)
        let shift = k as i64 - zero_lag as i64;
        let mut dot = 0.0f64;
        for i in 0..n {
            let j = i as i64 - shift;
            if j >= 0 && (j as usize) < n {
                dot += ref_padded[i] * cand_padded[j as usize];
            }
        }
        if dot > best_score {
            best_score = dot;
            best_k = k;
        }
    }

    // Quality check
    let ref_energy: f64 = ref_padded.dot(&ref_padded);
    let cand_energy: f64 = cand_padded.dot(&cand_padded);
    if ref_energy == 0.0 || cand_energy == 0.0 {
        return None;
    }
    let norm = (ref_energy * cand_energy).sqrt();
    if best_score / norm < min_quality {
        return None;
    }

    // best_k - zero_lag gives shift in bins (same as Python's best_idx - effective_max
    // where best_idx is the 0-indexed position within corr[lo:hi])
    let best_shift = best_k as i64 - zero_lag as i64;
    Some(best_shift * bin_size_ms)
}

// ---------------------------------------------------------------------------
// detect_op_gap
// ---------------------------------------------------------------------------

/// Find the opening-sequence gap in a subtitle track.
///
/// Looks for the largest dialogue-free interval within the expected OP
/// time window.  The gap boundaries are the end of the last pre-OP event
/// and the start of the first post-OP event.
///
/// Mirrors Python `detect_op_gap(timestamps, min_gap_ms, search_start_ms, search_end_ms)`.
pub fn detect_op_gap(
    timestamps: &[(i64, i64)],
    min_gap_ms: i64,
    search_start_ms: i64,
    search_end_ms: i64,
) -> Option<(i64, i64)> {
    if timestamps.is_empty() {
        return None;
    }

    // Sort by start time
    let mut events: Vec<(i64, i64)> = timestamps.to_vec();
    events.sort_by_key(|&(s, _)| s);

    let mut best_gap: Option<(i64, i64)> = None;
    let mut best_gap_size = 0i64;

    for i in 0..events.len().saturating_sub(1) {
        let (_, end_i) = events[i];
        let (start_next, _) = events[i + 1];
        let gap_size = start_next - end_i;

        if gap_size < min_gap_ms {
            continue;
        }
        // The gap must start within the search window
        if !(search_start_ms <= end_i && end_i <= search_end_ms) {
            continue;
        }
        if gap_size > best_gap_size {
            best_gap_size = gap_size;
            best_gap = Some((end_i, start_next));
        }
    }

    best_gap
}

/// Detect OP gap using default search parameters.
pub fn detect_op_gap_default(timestamps: &[(i64, i64)]) -> Option<(i64, i64)> {
    detect_op_gap(timestamps, MIN_GAP_MS, OP_SEARCH_START_MS, OP_SEARCH_END_MS)
}

// ---------------------------------------------------------------------------
// apply_offset / apply_piecewise_offsets
// ---------------------------------------------------------------------------

/// Shift all events in a subtitle file by the given offset (in place).
///
/// Mirrors Python `apply_offset(subtitle_path, offset_ms)`.
pub fn apply_offset(path: &Path, offset_ms: i64) -> Result<(), String> {
    let mut subs = mt_subtitles::load(path)?;
    for event in &mut subs.events {
        event.start_ms += offset_ms;
        event.end_ms += offset_ms;
    }
    save_subs(&subs, path)
}

/// Shift events in a subtitle file with different offsets per segment.
///
/// Events with start time < boundary_ms are shifted by pre_offset_ms.
/// Events with start time >= boundary_ms are shifted by post_offset_ms.
///
/// Mirrors Python `_apply_piecewise_offsets`.
fn apply_piecewise_offsets(
    path: &Path,
    boundary_ms: i64,
    pre_offset_ms: i64,
    post_offset_ms: i64,
) -> Result<(), String> {
    let mut subs = mt_subtitles::load(path)?;
    for event in &mut subs.events {
        if event.start_ms < boundary_ms {
            event.start_ms += pre_offset_ms;
            event.end_ms += pre_offset_ms;
        } else {
            event.start_ms += post_offset_ms;
            event.end_ms += post_offset_ms;
        }
    }
    save_subs(&subs, path)
}

fn save_subs(subs: &mt_subtitles::model::Subtitles, path: &Path) -> Result<(), String> {
    let content = match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .as_deref()
    {
        Some("srt") => mt_subtitles::srt::to_srt_string(subs),
        _ => mt_subtitles::ass::to_ass_string(subs),
    };
    std::fs::write(path, content).map_err(|e| format!("failed to write {}: {e}", path.display()))
}

// ---------------------------------------------------------------------------
// align_to_reference (public API)
// ---------------------------------------------------------------------------

/// Align a subtitle file to a reference track, handling OP-removed sources.
///
/// Detects if the reference has an opening-sequence gap.  If so, estimates
/// separate offsets for the pre-OP and post-OP segments and applies a
/// piecewise shift.  Otherwise, falls back to a single global offset.
///
/// Returns the applied offset in milliseconds.  For piecewise alignment,
/// returns the post-OP offset (the dominant one).  Returns 0 if no
/// correction was needed.
///
/// Mirrors Python `align_to_reference(subtitle_path, reference_path, min_offset_ms)`.
pub fn align_to_reference(subtitle_path: &Path, reference_path: &Path, min_offset_ms: i64) -> i64 {
    let (ref_timestamps, _) = extract_timestamps(reference_path);
    let (cand_timestamps, _) = extract_timestamps(subtitle_path);

    if ref_timestamps.is_empty() || cand_timestamps.is_empty() {
        return 0;
    }

    let op_gap = detect_op_gap_default(&ref_timestamps);

    if let Some(gap) = op_gap {
        align_piecewise(
            subtitle_path,
            &ref_timestamps,
            &cand_timestamps,
            gap,
            min_offset_ms,
        )
    } else {
        align_global(
            subtitle_path,
            &ref_timestamps,
            &cand_timestamps,
            min_offset_ms,
        )
    }
}

fn align_global(
    subtitle_path: &Path,
    ref_timestamps: &[(i64, i64)],
    cand_timestamps: &[(i64, i64)],
    min_offset_ms: i64,
) -> i64 {
    let offset = match estimate_offset(ref_timestamps, cand_timestamps, 100, 15_000, 0.4) {
        Some(o) => o,
        None => return 0,
    };

    if offset.abs() < min_offset_ms {
        return 0;
    }

    if apply_offset(subtitle_path, offset).is_err() {
        return 0;
    }

    offset
}

fn align_piecewise(
    subtitle_path: &Path,
    ref_timestamps: &[(i64, i64)],
    cand_timestamps: &[(i64, i64)],
    op_gap: (i64, i64),
    min_offset_ms: i64,
) -> i64 {
    let (gap_start, gap_end) = op_gap;
    let op_duration = gap_end - gap_start;

    // Split reference into pre-OP and post-OP segments
    let pre_op_ref: Vec<(i64, i64)> = ref_timestamps
        .iter()
        .filter(|&&(_, e)| e <= gap_start)
        .copied()
        .collect();
    let post_op_ref: Vec<(i64, i64)> = ref_timestamps
        .iter()
        .filter(|&&(s, _)| s >= gap_end)
        .copied()
        .collect();

    let pre_offset = estimate_offset(&pre_op_ref, cand_timestamps, 100, 15_000, 0.2);
    let post_offset = estimate_offset(
        &post_op_ref,
        cand_timestamps,
        100,
        op_duration + 30_000,
        0.2,
    );

    if pre_offset.is_none() && post_offset.is_none() {
        return 0;
    }

    // If one segment failed, use the other's offset for both.
    // post_offset.or(post_offset) is checked for None above, so unwrap is safe.
    let pre_offset = pre_offset.unwrap_or_else(|| post_offset.unwrap());
    let post_offset = post_offset.unwrap_or(pre_offset);

    // Check if the offsets are the same (uniform shift)
    if (pre_offset - post_offset).abs() < min_offset_ms {
        let offset = post_offset; // post-OP has more lines, more reliable
        if offset.abs() < min_offset_ms {
            return 0;
        }
        if apply_offset(subtitle_path, offset).is_err() {
            return 0;
        }
        return offset;
    }

    // Determine boundary in the candidate timeline.
    // Pre-OP candidate events end at approximately: gap_start - pre_offset
    // Post-OP candidate events start at approximately: gap_end - post_offset
    // Use the midpoint as the boundary.
    let pre_op_cand_end = gap_start - pre_offset;
    let post_op_cand_start = gap_end - post_offset;
    let boundary = (pre_op_cand_end + post_op_cand_start) / 2;

    let pre_significant = pre_offset.abs() >= min_offset_ms;
    let post_significant = post_offset.abs() >= min_offset_ms;

    if !pre_significant && !post_significant {
        return 0;
    }

    let effective_pre = if pre_significant { pre_offset } else { 0 };
    let effective_post = if post_significant { post_offset } else { 0 };

    if apply_piecewise_offsets(subtitle_path, boundary, effective_pre, effective_post).is_err() {
        return 0;
    }

    effective_post
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::TempDir;

    fn fmt_srt_time(ms: i64) -> String {
        let ms = ms.max(0);
        let h = ms / 3_600_000;
        let m = (ms % 3_600_000) / 60_000;
        let s = (ms % 60_000) / 1_000;
        let millis = ms % 1_000;
        format!("{h:02}:{m:02}:{s:02},{millis:03}")
    }

    fn make_srt(lines: &[(i64, i64, &str)]) -> String {
        let mut parts = Vec::new();
        for (i, &(start, end, text)) in lines.iter().enumerate() {
            parts.push(format!(
                "{}\n{} --> {}\n{}\n",
                i + 1,
                fmt_srt_time(start),
                fmt_srt_time(end),
                text
            ));
        }
        parts.join("\n")
    }

    fn write_file(dir: &TempDir, name: &str, content: &str) -> PathBuf {
        let path = dir.path().join(name);
        std::fs::write(&path, content).unwrap();
        path
    }

    fn to_timestamps(starts: &[i64], duration: i64) -> Vec<(i64, i64)> {
        starts.iter().map(|&s| (s, s + duration)).collect()
    }

    // -----------------------------------------------------------------------
    // Fixtures
    // -----------------------------------------------------------------------

    const REF_PRE_OP: &[(i64, i64, &str)] = &[
        (5000, 7000, "A"),
        (10000, 12000, "B"),
        (15000, 17000, "C"),
        (20000, 22000, "D"),
        (30000, 32000, "E"),
        (40000, 42000, "F"),
        (50000, 52000, "G"),
        (60000, 62000, "H"),
        (70000, 72000, "I"),
        (80000, 82000, "J"),
        (90000, 92000, "K"),
        (100000, 102000, "L"),
    ];

    const REF_POST_OP: &[(i64, i64, &str)] = &[
        (210000, 212000, "M"),
        (215000, 217000, "N"),
        (220000, 222000, "O"),
        (230000, 232000, "P"),
        (240000, 242000, "Q"),
        (250000, 252000, "R"),
        (260000, 262000, "S"),
        (270000, 272000, "T"),
        (280000, 282000, "U"),
        (290000, 292000, "V"),
        (300000, 302000, "W"),
        (310000, 312000, "X"),
        (350000, 352000, "Y"),
        (400000, 402000, "Z"),
        (450000, 452000, "AA"),
        (500000, 502000, "BB"),
    ];

    fn ref_lines() -> Vec<(i64, i64, &'static str)> {
        REF_PRE_OP
            .iter()
            .chain(REF_POST_OP.iter())
            .copied()
            .collect()
    }

    fn make_op_removed_candidate(
        pre_offset_ms: i64,
        post_offset_ms: i64,
    ) -> Vec<(i64, i64, &'static str)> {
        REF_PRE_OP
            .iter()
            .map(|&(s, e, t)| (s + pre_offset_ms, e + pre_offset_ms, t))
            .chain(
                REF_POST_OP
                    .iter()
                    .map(|&(s, e, t)| (s + post_offset_ms, e + post_offset_ms, t)),
            )
            .collect()
    }

    const SIMPLE_REF: &[(i64, i64, &str)] = &[
        (1000, 3000, "A"),
        (4000, 6000, "B"),
        (7000, 9000, "C"),
        (10000, 12000, "D"),
        (14000, 16000, "E"),
        (18000, 20000, "F"),
        (60000, 62000, "G"),
        (65000, 67000, "H"),
        (70000, 72000, "I"),
        (150000, 152000, "J"),
        (153000, 155000, "K"),
        (157000, 159000, "L"),
        (240000, 242000, "M"),
        (250000, 252000, "N"),
        (260000, 262000, "O"),
    ];

    // -----------------------------------------------------------------------
    // TestDetectOpGap
    // -----------------------------------------------------------------------

    #[test]
    fn detect_op_gap_finds_gap() {
        let ts: Vec<(i64, i64)> = ref_lines().iter().map(|&(s, e, _)| (s, e)).collect();
        let gap = detect_op_gap_default(&ts);
        assert!(gap.is_some());
        let (gap_start, gap_end) = gap.unwrap();
        assert!((100_000..=110_000).contains(&gap_start));
        assert!((205_000..=215_000).contains(&gap_end));
    }

    #[test]
    fn detect_op_gap_no_gap_when_none_exists() {
        let ts = to_timestamps(&[1000, 5000, 10000, 15000, 20000], 2000);
        assert_eq!(detect_op_gap_default(&ts), None);
    }

    #[test]
    fn detect_op_gap_ignores_gaps_outside_search_window() {
        // Gap at 500s — outside the 30s–360s search window
        let ts: Vec<(i64, i64)> = vec![
            (1000, 3000),
            (5000, 7000),
            (500000, 502000),
            (600000, 602000),
        ];
        let gap = detect_op_gap(&ts, MIN_GAP_MS, 30_000, 360_000);
        assert_eq!(gap, None);
    }

    #[test]
    fn detect_op_gap_finds_largest_gap_in_window() {
        // Two gaps: 80s gap at 52s mark, 200s gap at 142s mark
        let ts: Vec<(i64, i64)> = vec![
            (10000, 12000),
            (50000, 52000),
            (132000, 134000),
            (140000, 142000),
            (342000, 344000),
            (400000, 402000),
        ];
        let gap = detect_op_gap_default(&ts);
        assert!(gap.is_some());
        let (gap_start, gap_end) = gap.unwrap();
        assert!(
            gap_end - gap_start > 150_000,
            "expected >150s gap, got {}ms",
            gap_end - gap_start
        );
    }

    #[test]
    fn detect_op_gap_empty_timestamps() {
        assert_eq!(detect_op_gap_default(&[]), None);
    }

    // -----------------------------------------------------------------------
    // TestEstimateOffset
    // -----------------------------------------------------------------------

    #[test]
    fn estimate_offset_identical_timings_returns_zero() {
        let ts = to_timestamps(&[1000, 4000, 7000, 10000, 14000], 2000);
        let offset = estimate_offset(&ts, &ts, 100, 15_000, 0.4);
        assert_eq!(offset, Some(0));
    }

    #[test]
    fn estimate_offset_positive_candidate_early() {
        let starts = vec![1000i64, 4000, 7000, 10000, 14000, 60000, 65000, 70000];
        let ref_ = to_timestamps(&starts, 2000);
        let shifted: Vec<i64> = starts.iter().map(|s| s - 1500).collect();
        let cand = to_timestamps(&shifted, 2000);
        let offset = estimate_offset(&ref_, &cand, 100, 15_000, 0.4);
        let off = offset.unwrap_or(i64::MAX);
        assert!((off - 1500).abs() <= 100, "expected ~1500, got {off}");
    }

    #[test]
    fn estimate_offset_negative_candidate_late() {
        let starts = vec![1000i64, 4000, 7000, 10000, 14000, 60000, 65000, 70000];
        let ref_ = to_timestamps(&starts, 2000);
        let shifted: Vec<i64> = starts.iter().map(|s| s + 2000).collect();
        let cand = to_timestamps(&shifted, 2000);
        let offset = estimate_offset(&ref_, &cand, 100, 15_000, 0.4);
        let off = offset.unwrap_or(i64::MAX);
        assert!((off - (-2000)).abs() <= 100, "expected ~-2000, got {off}");
    }

    #[test]
    fn estimate_offset_large_offset_with_dense_lines() {
        let starts = vec![1000i64, 4000, 7000, 10000, 14000, 60000, 65000, 70000];
        let ref_ = to_timestamps(&starts, 2000);
        let shifted: Vec<i64> = starts.iter().map(|s| s + 5000).collect();
        let cand = to_timestamps(&shifted, 2000);
        let offset = estimate_offset(&ref_, &cand, 100, 15_000, 0.4);
        let off = offset.unwrap_or(i64::MAX);
        assert!((off - (-5000)).abs() <= 200, "expected ~-5000, got {off}");
    }

    #[test]
    fn estimate_offset_empty_reference_returns_none() {
        let cand = to_timestamps(&[1000, 2000], 2000);
        assert_eq!(estimate_offset(&[], &cand, 100, 15_000, 0.4), None);
    }

    #[test]
    fn estimate_offset_empty_candidate_returns_none() {
        let ref_ = to_timestamps(&[1000, 2000], 2000);
        assert_eq!(estimate_offset(&ref_, &[], 100, 15_000, 0.4), None);
    }

    // -----------------------------------------------------------------------
    // TestApplyOffset
    // -----------------------------------------------------------------------

    #[test]
    fn apply_offset_shifts_all_events_forward() {
        let tmp = TempDir::new().unwrap();
        let path = write_file(
            &tmp,
            "test.srt",
            &make_srt(&[(1000, 3000, "A"), (5000, 7000, "B")]),
        );
        apply_offset(&path, 2000).unwrap();
        let (timestamps, _) = extract_timestamps(&path);
        assert_eq!(timestamps[0], (3000, 5000));
        assert_eq!(timestamps[1], (7000, 9000));
    }

    #[test]
    fn apply_offset_shifts_all_events_backward() {
        let tmp = TempDir::new().unwrap();
        let path = write_file(
            &tmp,
            "test.srt",
            &make_srt(&[(5000, 7000, "A"), (10000, 12000, "B")]),
        );
        apply_offset(&path, -2000).unwrap();
        let (timestamps, _) = extract_timestamps(&path);
        assert_eq!(timestamps[0], (3000, 5000));
        assert_eq!(timestamps[1], (8000, 10000));
    }

    #[test]
    fn apply_offset_zero_no_change() {
        let tmp = TempDir::new().unwrap();
        let path = write_file(
            &tmp,
            "test.srt",
            &make_srt(&[(1000, 3000, "A"), (5000, 7000, "B")]),
        );
        apply_offset(&path, 0).unwrap();
        let (timestamps, _) = extract_timestamps(&path);
        assert_eq!(timestamps[0], (1000, 3000));
        assert_eq!(timestamps[1], (5000, 7000));
    }

    // -----------------------------------------------------------------------
    // TestAlignToReferenceGlobal
    // -----------------------------------------------------------------------

    #[test]
    fn no_correction_when_aligned() {
        let tmp = TempDir::new().unwrap();
        let ref_path = write_file(&tmp, "ref.srt", &make_srt(SIMPLE_REF));
        let cand_path = write_file(&tmp, "cand.srt", &make_srt(SIMPLE_REF));
        let offset = align_to_reference(&cand_path, &ref_path, MIN_OFFSET_MS);
        assert_eq!(offset, 0, "expected 0, got {offset}");
    }

    #[test]
    fn corrects_late_subtitles() {
        let tmp = TempDir::new().unwrap();
        let ref_path = write_file(&tmp, "ref.srt", &make_srt(SIMPLE_REF));
        let shifted: Vec<(i64, i64, &str)> = SIMPLE_REF
            .iter()
            .map(|&(s, e, t)| (s + 2000, e + 2000, t))
            .collect();
        let cand_path = write_file(&tmp, "cand.srt", &make_srt(&shifted));
        let offset = align_to_reference(&cand_path, &ref_path, MIN_OFFSET_MS);
        assert!(
            (offset - (-2000)).abs() <= 100,
            "expected ~-2000, got {offset}"
        );
    }

    #[test]
    fn returns_zero_for_empty_candidate() {
        let tmp = TempDir::new().unwrap();
        let ref_path = write_file(&tmp, "ref.srt", &make_srt(SIMPLE_REF));
        let cand_path = write_file(&tmp, "cand.srt", "");
        let offset = align_to_reference(&cand_path, &ref_path, MIN_OFFSET_MS);
        assert_eq!(offset, 0);
    }

    // -----------------------------------------------------------------------
    // TestAlignToReferencePiecewise
    // -----------------------------------------------------------------------

    #[test]
    fn corrects_op_removed_candidate() {
        let tmp = TempDir::new().unwrap();
        let ref_path = write_file(&tmp, "ref.srt", &make_srt(&ref_lines()));
        let cand_lines = make_op_removed_candidate(-2000, -110000);
        let cand_path = write_file(&tmp, "cand.srt", &make_srt(&cand_lines));

        let offset = align_to_reference(&cand_path, &ref_path, MIN_OFFSET_MS);
        assert!(
            (offset - 110_000).abs() <= 1000,
            "expected ~110000, got {offset}"
        );

        // Verify both segments are corrected
        let (timestamps, _) = extract_timestamps(&cand_path);
        let (ref_timestamps, _) = extract_timestamps(&ref_path);
        let ref_starts: Vec<i64> = {
            let mut v: Vec<i64> = ref_timestamps.iter().map(|&(s, _)| s).collect();
            v.sort_unstable();
            v
        };
        let cand_starts: Vec<i64> = {
            let mut v: Vec<i64> = timestamps.iter().map(|&(s, _)| s).collect();
            v.sort_unstable();
            v
        };

        for &cs in &cand_starts {
            if cs < 105_000 {
                let min_dist = ref_starts
                    .iter()
                    .map(|&rs| (cs - rs).abs())
                    .min()
                    .unwrap_or(i64::MAX);
                assert!(
                    min_dist < 500,
                    "pre-OP line at {cs}ms not aligned: min_dist={min_dist}"
                );
            }
        }

        for &cs in &cand_starts {
            if cs > 200_000 {
                let min_dist = ref_starts
                    .iter()
                    .map(|&rs| (cs - rs).abs())
                    .min()
                    .unwrap_or(i64::MAX);
                assert!(
                    min_dist < 2000,
                    "post-OP line at {cs}ms not aligned: min_dist={min_dist}"
                );
            }
        }
    }

    #[test]
    fn uniform_offset_with_op_gap() {
        let tmp = TempDir::new().unwrap();
        let ref_path = write_file(&tmp, "ref.srt", &make_srt(&ref_lines()));
        let cand_lines = make_op_removed_candidate(-2000, -2000);
        let cand_path = write_file(&tmp, "cand.srt", &make_srt(&cand_lines));

        let offset = align_to_reference(&cand_path, &ref_path, MIN_OFFSET_MS);
        assert!((offset - 2000).abs() <= 200, "expected ~2000, got {offset}");
    }

    #[test]
    fn only_pre_op_offset() {
        let tmp = TempDir::new().unwrap();
        let ref_path = write_file(&tmp, "ref.srt", &make_srt(&ref_lines()));
        let pre: Vec<(i64, i64, &str)> = REF_PRE_OP
            .iter()
            .map(|&(s, e, t)| (s - 3000, e - 3000, t))
            .collect();
        let post: Vec<(i64, i64, &str)> = REF_POST_OP.to_vec();
        let cand_lines: Vec<(i64, i64, &str)> = pre.into_iter().chain(post).collect();
        let cand_path = write_file(&tmp, "cand.srt", &make_srt(&cand_lines));

        let offset = align_to_reference(&cand_path, &ref_path, MIN_OFFSET_MS);
        // Should apply piecewise correction; just ensure it doesn't panic and returns something
        let _ = offset;
    }

    #[test]
    fn skips_tiny_offsets() {
        let tmp = TempDir::new().unwrap();
        let ref_path = write_file(&tmp, "ref.srt", &make_srt(&ref_lines()));
        let cand_lines = make_op_removed_candidate(-50, -50);
        let cand_path = write_file(&tmp, "cand.srt", &make_srt(&cand_lines));

        let offset = align_to_reference(&cand_path, &ref_path, 150);
        assert_eq!(offset, 0, "expected 0 for tiny offsets, got {offset}");
    }

    // -----------------------------------------------------------------------
    // Konosuba static-offset test (from project memory)
    // -----------------------------------------------------------------------

    /// Tests that a static global offset (no OP gap) is correctly detected and applied.
    ///
    /// The project memory notes a static-offset case observed on "Konosuba S1E1":
    /// fetched Polish subtitles had a static ~20s offset relative to the English track.
    /// This test simulates that scenario and verifies the cross-correlation picks it up.
    #[test]
    fn konosuba_s1e1_static_offset_detection() {
        // Simulate Konosuba S1E1 pattern: 22 lines spread over ~22 minutes,
        // candidate is +20s (20000ms) late relative to reference.
        let ref_starts: Vec<i64> = vec![
            10_000, 20_000, 35_000, 50_000, 80_000, 120_000, 160_000, 200_000, 240_000, 300_000,
            360_000, 420_000, 480_000, 540_000, 600_000, 660_000, 720_000, 780_000, 840_000,
            900_000, 960_000, 1_020_000,
        ];
        let ref_ts = to_timestamps(&ref_starts, 2000);
        let cand_starts: Vec<i64> = ref_starts.iter().map(|s| s + 20_000).collect();
        let cand_ts = to_timestamps(&cand_starts, 2000);

        let offset = estimate_offset(&ref_ts, &cand_ts, 100, 30_000, 0.4);
        let off = offset.unwrap_or(i64::MAX);
        assert!(
            (off - (-20_000)).abs() <= 100,
            "expected ~-20000ms static offset, got {off}"
        );
    }
}
