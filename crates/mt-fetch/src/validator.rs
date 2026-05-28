//! Subtitle validation via line-level timing matching.
//!
//! Compares downloaded subtitle candidates against a reference track by
//! matching individual dialogue line start times.  For each candidate line,
//! finds the nearest reference line and checks if it falls within a tolerance
//! window.  The fraction of matched lines is the similarity score.

use std::path::Path;

use mt_subtitles::model::Event;
use ndarray::Array1;

use crate::retry::FetchError;
use crate::style_classifier::classify_styles;
use crate::types::SubtitleMatch;

// ---------------------------------------------------------------------------
// build_activity_vector
// ---------------------------------------------------------------------------

/// Convert subtitle timestamps to a binary activity vector.
///
/// Divides the timeline into fixed-width bins and marks each bin as 1
/// if any dialogue event overlaps it, 0 otherwise.
pub fn build_activity_vector(
    timestamps: &[(i64, i64)],
    duration_ms: i64,
    bin_size_ms: i64,
) -> Array1<f64> {
    let n_bins = if duration_ms > 0 {
        ((duration_ms as f64) / (bin_size_ms as f64)).ceil() as usize
    } else {
        0
    };

    if n_bins == 0 {
        return Array1::zeros(0);
    }

    let mut vec = Array1::<f64>::zeros(n_bins);

    for &(start, end) in timestamps {
        let first_bin = (start / bin_size_ms).max(0) as usize;
        // Last bin that the event touches (inclusive).
        // An event ending exactly on a bin boundary doesn't spill into the next bin.
        let last_bin = if end > start {
            ((end - 1) / bin_size_ms).min((n_bins as i64) - 1) as usize
        } else {
            first_bin.min(n_bins - 1)
        };
        for b in first_bin..=last_bin {
            vec[b] = 1.0;
        }
    }

    vec
}

// ---------------------------------------------------------------------------
// compute_similarity
// ---------------------------------------------------------------------------

/// Compute normalized cross-correlation between two activity vectors.
///
/// Tries shifts from -max_shift_bins to +max_shift_bins and returns the
/// peak correlation, normalised by the geometric mean of energies.
pub fn compute_similarity(
    reference: &Array1<f64>,
    candidate: &Array1<f64>,
    max_shift_bins: usize,
) -> f64 {
    let ref_energy: f64 = reference.dot(reference);
    let cand_energy: f64 = candidate.dot(candidate);

    if ref_energy == 0.0 || cand_energy == 0.0 {
        return 0.0;
    }

    let norm = (ref_energy * cand_energy).sqrt();

    // Pad both vectors to the same length.
    let max_len = reference.len().max(candidate.len());
    let mut ref_padded = Array1::<f64>::zeros(max_len);
    let mut cand_padded = Array1::<f64>::zeros(max_len);
    ref_padded
        .slice_mut(ndarray::s![..reference.len()])
        .assign(reference);
    cand_padded
        .slice_mut(ndarray::s![..candidate.len()])
        .assign(candidate);

    // Clamp shift range so we don't exceed vector length.
    let effective_max = max_shift_bins.min(max_len.saturating_sub(1));

    // Compute full cross-correlation (mode='full'):  corr[k] = sum_i ref[i] * cand[i - (k - (n-1))]
    // zero_lag = len(ref) - 1
    let n = max_len;
    let full_len = 2 * n - 1;
    let zero_lag = n - 1;
    let lo = zero_lag.saturating_sub(effective_max);
    let hi = (zero_lag + effective_max + 1).min(full_len);

    let mut best: f64 = f64::NEG_INFINITY;
    for k in lo..hi {
        // corr[k] = sum over i of ref[i] * cand[i - (k - zero_lag)]
        // shift = k - zero_lag
        let shift = k as i64 - zero_lag as i64;
        let mut dot = 0.0f64;
        for i in 0..n {
            let j = i as i64 - shift;
            if j >= 0 && (j as usize) < n {
                dot += ref_padded[i] * cand_padded[j as usize];
            }
        }
        if dot > best {
            best = dot;
        }
    }

    let score = best / norm;
    score.min(1.0)
}

// ---------------------------------------------------------------------------
// compute_line_match_score
// ---------------------------------------------------------------------------

/// Score how well candidate line timings match reference line timings.
///
/// For each candidate line start time, finds the nearest reference line
/// start time using binary search.  A candidate line is "matched" if the
/// nearest reference line is within `tolerance_ms`.  The score is the
/// fraction of candidate lines that matched.
pub fn compute_line_match_score(ref_starts: &[i64], cand_starts: &[i64], tolerance_ms: i64) -> f64 {
    if ref_starts.is_empty() || cand_starts.is_empty() {
        return 0.0;
    }

    let mut matched = 0usize;
    for &cand_t in cand_starts {
        // Binary search for insertion point
        let idx = ref_starts.partition_point(|&x| x < cand_t);
        let mut best_dist = i64::MAX;

        // Check insertion point and its left neighbour
        for &i in &[idx.wrapping_sub(1), idx] {
            if i < ref_starts.len() {
                let dist = (ref_starts[i] - cand_t).abs();
                if dist < best_dist {
                    best_dist = dist;
                }
            }
        }

        if best_dist <= tolerance_ms {
            matched += 1;
        }
    }

    matched as f64 / cand_starts.len() as f64
}

// ---------------------------------------------------------------------------
// build_density_vector
// ---------------------------------------------------------------------------

/// Build a dialogue density vector — count of events starting in each window.
pub fn build_density_vector(
    timestamps: &[(i64, i64)],
    duration_ms: i64,
    window_ms: i64,
) -> Array1<f64> {
    let n_bins = if duration_ms > 0 {
        ((duration_ms as f64) / (window_ms as f64)).ceil() as usize
    } else {
        0
    };

    if n_bins == 0 {
        return Array1::zeros(0);
    }

    let mut vec = Array1::<f64>::zeros(n_bins);

    for &(start, _) in timestamps {
        let bin_idx = ((start / window_ms) as usize).min(n_bins - 1);
        vec[bin_idx] += 1.0;
    }

    vec
}

// ---------------------------------------------------------------------------
// compute_density_correlation
// ---------------------------------------------------------------------------

/// Compute Pearson correlation between density vectors with shifting.
pub fn compute_density_correlation(
    ref_density: &Array1<f64>,
    cand_density: &Array1<f64>,
    max_shift: usize,
) -> f64 {
    if ref_density.is_empty() || cand_density.is_empty() {
        return 0.0;
    }

    // Pad to same length
    let max_len = ref_density.len().max(cand_density.len());
    let mut ref_padded = Array1::<f64>::zeros(max_len);
    let mut cand_padded = Array1::<f64>::zeros(max_len);
    ref_padded
        .slice_mut(ndarray::s![..ref_density.len()])
        .assign(ref_density);
    cand_padded
        .slice_mut(ndarray::s![..cand_density.len()])
        .assign(cand_density);

    let ref_std = std_dev(&ref_padded);
    let cand_std = std_dev(&cand_padded);

    if ref_std == 0.0 || cand_std == 0.0 {
        return 0.0;
    }

    let effective_max = max_shift.min(max_len.saturating_sub(1));
    let mut best = 0.0f64;

    for shift_abs in -(effective_max as i64)..=(effective_max as i64) {
        let (r_slice, c_slice) = if shift_abs >= 0 {
            let s = shift_abs as usize;
            let r = ref_padded.slice(ndarray::s![s..]).to_owned();
            let c = cand_padded.slice(ndarray::s![..max_len - s]).to_owned();
            (r, c)
        } else {
            let s = (-shift_abs) as usize;
            let r = ref_padded.slice(ndarray::s![..max_len - s]).to_owned();
            let c = cand_padded.slice(ndarray::s![s..]).to_owned();
            (r, c)
        };

        if r_slice.len() < 3 {
            continue;
        }

        let r_mean = r_slice.mean().unwrap_or(0.0);
        let c_mean = c_slice.mean().unwrap_or(0.0);
        let r_std_s = std_dev(&r_slice);
        let c_std_s = std_dev(&c_slice);

        if r_std_s == 0.0 || c_std_s == 0.0 {
            continue;
        }

        let len = r_slice.len() as f64;
        let corr: f64 = r_slice
            .iter()
            .zip(c_slice.iter())
            .map(|(&ri, &ci)| (ri - r_mean) * (ci - c_mean))
            .sum::<f64>()
            / (len * r_std_s * c_std_s);

        if corr > best {
            best = corr;
        }
    }

    best
}

fn std_dev(arr: &Array1<f64>) -> f64 {
    if arr.is_empty() {
        return 0.0;
    }
    let mean = arr.mean().unwrap_or(0.0);
    let variance = arr.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / arr.len() as f64;
    variance.sqrt()
}

// ---------------------------------------------------------------------------
// extract_timestamps
// ---------------------------------------------------------------------------

/// Extract dialogue timestamps from a parsed subtitle event list.
///
/// Filters out non-dialogue events (signs, songs, etc.) using the style
/// classifier and keyword filter, then returns timing pairs.
///
/// Accepts pre-parsed events to decouple I/O. The public
/// [`extract_timestamps_from_path`] wraps this.
pub fn extract_timestamps_from_events(events: &[Event]) -> (Vec<(i64, i64)>, i64) {
    // Use structural classification as primary filter.
    let dialogue_styles = classify_styles(events);

    let mut timestamps: Vec<(i64, i64)> = Vec::new();

    for event in events {
        if event.text.trim().is_empty() {
            continue;
        }

        // Primary: structural classifier
        if !dialogue_styles.contains(&event.style) {
            continue;
        }

        // Secondary: keyword filter (catches song lyrics that look like dialogue).
        // Use the shared whole-token matcher so styles like "Top"/"Named" are
        // not misclassified by a bare substring match on "op"/"ed".
        if mt_core::is_non_dialogue_style(&event.style) {
            continue;
        }

        // Skip empty plaintext (after stripping ASS tags)
        let plain = event.plaintext();
        if plain.trim().is_empty() {
            continue;
        }

        timestamps.push((event.start_ms, event.end_ms));
    }

    let duration_ms = timestamps.iter().map(|&(_, e)| e).max().unwrap_or(0);
    (timestamps, duration_ms)
}

/// Extract dialogue timestamps from a subtitle file on disk.
pub fn extract_timestamps(path: &Path) -> (Vec<(i64, i64)>, i64) {
    let subs = match mt_subtitles::load(path) {
        Ok(s) => s,
        Err(_) => return (vec![], 0),
    };
    extract_timestamps_from_events(&subs.events)
}

/// Like [`extract_timestamps`] but surfaces parse/IO failures instead of
/// silently returning an empty result.
pub fn extract_timestamps_checked(path: &Path) -> Result<(Vec<(i64, i64)>, i64), FetchError> {
    let subs = mt_subtitles::load(path)
        .map_err(|e| FetchError::Parse(format!("{}: {e}", path.display())))?;
    Ok(extract_timestamps_from_events(&subs.events))
}

// ---------------------------------------------------------------------------
// SubtitleValidator
// ---------------------------------------------------------------------------

/// Validates subtitle candidates against a reference track.
///
/// Uses line-level timing matching: for each candidate dialogue line,
/// finds the nearest reference line by start time and checks if it falls
/// within a tolerance window.  The fraction of matched lines is the score.
pub struct SubtitleValidator {
    pub ref_timestamps: Vec<(i64, i64)>,
    pub ref_duration: i64,
    pub window_ms: i64,
}

impl SubtitleValidator {
    /// Create a new validator from a reference subtitle file.
    pub fn new(reference_path: &Path, window_ms: i64) -> Self {
        let (ref_timestamps, ref_duration) = extract_timestamps(reference_path);
        Self {
            ref_timestamps,
            ref_duration,
            window_ms,
        }
    }

    /// Score a single candidate subtitle file against the reference.
    ///
    /// Returns an error if the candidate file cannot be read/parsed, so callers
    /// can skip it via normal control flow.
    pub fn score_candidate(&self, candidate_path: &Path) -> Result<f64, FetchError> {
        let cand_timestamps = extract_timestamps_checked(candidate_path)?.0;

        if cand_timestamps.is_empty() || self.ref_timestamps.is_empty() {
            return Ok(0.0);
        }

        let mut ref_starts: Vec<i64> = self.ref_timestamps.iter().map(|&(s, _)| s).collect();
        ref_starts.sort_unstable();
        let mut cand_starts: Vec<i64> = cand_timestamps.iter().map(|&(s, _)| s).collect();
        cand_starts.sort_unstable();

        Ok(compute_line_match_score(&ref_starts, &cand_starts, 2000))
    }

    /// Score all candidates, filter by threshold, sort by score descending.
    pub fn validate_candidates(
        &self,
        candidates: &[(SubtitleMatch, std::path::PathBuf)],
        min_threshold: f64,
    ) -> Vec<(SubtitleMatch, std::path::PathBuf, f64)> {
        let mut results = Vec::new();

        for (match_, path) in candidates {
            let score = match self.score_candidate(path) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!("Failed to score candidate {}: {e}", match_.subtitle_id);
                    continue;
                }
            };

            if score >= min_threshold {
                results.push((match_.clone(), path.clone(), score));
            }
        }

        results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        results
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::TempDir;

    // Helper: write a file and return its path
    fn write_file(dir: &TempDir, name: &str, content: &str) -> PathBuf {
        let path = dir.path().join(name);
        std::fs::write(&path, content).unwrap();
        path
    }

    // -----------------------------------------------------------------------
    // SRT helpers
    // -----------------------------------------------------------------------

    fn fmt_srt_time(ms: i64) -> String {
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

    // -----------------------------------------------------------------------
    // TestBuildActivityVector
    // -----------------------------------------------------------------------

    #[test]
    fn empty_timestamps_returns_all_zeros() {
        let vec = build_activity_vector(&[], 10000, 2000);
        assert_eq!(vec.len(), 5);
        for v in vec.iter() {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn single_event_marks_correct_bins() {
        // Event from 1000-3000ms with 2000ms bins covers bins 0 and 1
        let vec = build_activity_vector(&[(1000, 3000)], 10000, 2000);
        assert_eq!(vec.len(), 5);
        assert_eq!(vec[0], 1.0);
        assert_eq!(vec[1], 1.0);
        assert_eq!(vec[2], 0.0);
        assert_eq!(vec[3], 0.0);
        assert_eq!(vec[4], 0.0);
    }

    #[test]
    fn event_spanning_all_bins() {
        let vec = build_activity_vector(&[(0, 10000)], 10000, 2000);
        for v in vec.iter() {
            assert_eq!(*v, 1.0);
        }
    }

    #[test]
    fn multiple_events() {
        // Two events: 0-2000 (bin 0) and 6000-8000 (bin 3)
        let timestamps = [(0, 2000), (6000, 8000)];
        let vec = build_activity_vector(&timestamps, 10000, 2000);
        assert_eq!(vec[0], 1.0);
        assert_eq!(vec[1], 0.0);
        assert_eq!(vec[2], 0.0);
        assert_eq!(vec[3], 1.0);
        assert_eq!(vec[4], 0.0);
    }

    #[test]
    fn event_partially_overlapping_bin() {
        // Event from 1999-2001 should overlap bins 0 and 1
        let vec = build_activity_vector(&[(1999, 2001)], 10000, 2000);
        assert_eq!(vec[0], 1.0);
        assert_eq!(vec[1], 1.0);
    }

    #[test]
    fn custom_bin_size() {
        let vec = build_activity_vector(&[(0, 500)], 2000, 1000);
        assert_eq!(vec.len(), 2);
        assert_eq!(vec[0], 1.0);
        assert_eq!(vec[1], 0.0);
    }

    #[test]
    fn duration_not_exact_multiple_of_bin_size() {
        // 7000ms / 2000ms = 3.5 -> should ceil to 4 bins
        let vec = build_activity_vector(&[], 7000, 2000);
        assert_eq!(vec.len(), 4);
    }

    #[test]
    fn event_at_very_end() {
        // Event in the last bin
        let vec = build_activity_vector(&[(8000, 10000)], 10000, 2000);
        assert_eq!(vec[4], 1.0);
        for i in 0..4 {
            assert_eq!(vec[i], 0.0);
        }
    }

    #[test]
    fn overlapping_events_still_binary() {
        // Two overlapping events should still produce 1, not 2
        let timestamps = [(0, 5000), (3000, 7000)];
        let vec = build_activity_vector(&timestamps, 10000, 2000);
        // Bins 0-3 should be active
        assert_eq!(vec[0], 1.0);
        assert_eq!(vec[1], 1.0);
        assert_eq!(vec[2], 1.0);
        assert_eq!(vec[3], 1.0);
        assert_eq!(vec[4], 0.0);
        // All values are 0 or 1 (binary)
        assert!(vec.iter().all(|&v| v == 0.0 || v == 1.0));
    }

    #[test]
    fn zero_duration_returns_empty() {
        let vec = build_activity_vector(&[], 0, 2000);
        assert_eq!(vec.len(), 0);
    }

    // -----------------------------------------------------------------------
    // TestComputeSimilarity
    // -----------------------------------------------------------------------

    #[test]
    fn identical_vectors_return_1() {
        let vec = Array1::from(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
        let score = compute_similarity(&vec, &vec, 15);
        assert!((score - 1.0).abs() < 1e-9, "expected 1.0 got {score}");
    }

    #[test]
    fn opposite_vectors_return_high_score() {
        // With shifting, the complement pattern shifted by 1 would match perfectly
        let ref_ = Array1::from(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
        let cand = Array1::from(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        let score = compute_similarity(&ref_, &cand, 15);
        assert!((score - 1.0).abs() < 1e-9, "expected 1.0 got {score}");
    }

    #[test]
    fn no_overlap_returns_zero() {
        let ref_ = Array1::from(vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let cand = Array1::zeros(10);
        let score = compute_similarity(&ref_, &cand, 15);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn shifted_vector_detected() {
        // Reference active in bins 0-4, candidate in bins 5-9
        let ref_ = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let cand = Array1::from(vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let score = compute_similarity(&ref_, &cand, 5);
        assert!((score - 1.0).abs() < 1e-9, "expected 1.0 got {score}");
    }

    #[test]
    fn shifted_beyond_max_returns_low() {
        // Shift of 5 needed but max_shift is 2
        let ref_ = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let cand = Array1::from(vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let score = compute_similarity(&ref_, &cand, 2);
        assert!(score < 0.5, "expected < 0.5 got {score}");
    }

    #[test]
    fn score_between_0_and_1() {
        // Use a deterministic "random" sequence
        let ref_vals: Vec<f64> = (0..100)
            .map(|i| if (i * 7 + 3) % 3 == 0 { 1.0 } else { 0.0 })
            .collect();
        let cand_vals: Vec<f64> = (0..100)
            .map(|i| if (i * 11 + 5) % 3 == 0 { 1.0 } else { 0.0 })
            .collect();
        let ref_ = Array1::from(ref_vals);
        let cand = Array1::from(cand_vals);
        let score = compute_similarity(&ref_, &cand, 15);
        assert!((0.0..=1.0).contains(&score), "expected 0..=1 got {score}");
    }

    #[test]
    fn both_empty_returns_zero() {
        let ref_: Array1<f64> = Array1::zeros(10);
        let cand: Array1<f64> = Array1::zeros(10);
        let score = compute_similarity(&ref_, &cand, 15);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn different_lengths_padded() {
        let ref_ = Array1::from(vec![1.0, 0.0, 1.0, 0.0, 1.0]);
        let cand = Array1::from(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let score = compute_similarity(&ref_, &cand, 15);
        assert!(score > 0.7, "expected > 0.7 got {score}");
    }

    #[test]
    fn partial_overlap_gives_intermediate_score() {
        let ref_ = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let cand = Array1::from(vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let score = compute_similarity(&ref_, &cand, 0);
        assert!(
            score > 0.0 && score < 1.0,
            "expected 0 < score < 1 got {score}"
        );
    }

    // -----------------------------------------------------------------------
    // TestExtractTimestamps — using actual SRT/ASS fixtures
    // -----------------------------------------------------------------------

    const SRT_CONTENT: &str = "\
1
00:00:01,000 --> 00:00:03,000
Hello, world!

2
00:00:05,000 --> 00:00:07,000
Second line.

3
00:00:10,000 --> 00:00:12,000
Third line.
";

    // ASS content with dialogue, signs, and song styles.
    const ASS_CONTENT: &str = "\
[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1
Style: Sign,Arial,36,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1
Style: OP,Arial,36,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1
Style: Song-Lyrics,Arial,36,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Hello from ASS
Dialogue: 0,0:00:05.00,0:00:07.00,Sign,,0,0,0,,{\\an8}Sign text
Dialogue: 0,0:00:08.00,0:00:10.00,Default,,0,0,0,,Second dialogue
Dialogue: 0,0:00:12.00,0:00:14.00,OP,,0,0,0,,Opening song
Dialogue: 0,0:00:15.00,0:00:17.00,Song-Lyrics,,0,0,0,,La la la
";

    #[test]
    fn srt_extracts_all_events() {
        let tmp = TempDir::new().unwrap();
        let path = write_file(&tmp, "test.srt", SRT_CONTENT);
        let (timestamps, duration) = extract_timestamps(&path);
        assert_eq!(timestamps.len(), 3);
        assert_eq!(timestamps[0], (1000, 3000));
        assert_eq!(timestamps[1], (5000, 7000));
        assert_eq!(timestamps[2], (10000, 12000));
        assert_eq!(duration, 12000);
    }

    #[test]
    fn ass_filters_non_dialogue_styles() {
        let tmp = TempDir::new().unwrap();
        let path = write_file(&tmp, "test.ass", ASS_CONTENT);
        let (timestamps, duration) = extract_timestamps(&path);
        // Only Default style events (OP style filtered by keyword, Sign has pos tag → style classifier)
        assert_eq!(timestamps.len(), 2, "timestamps: {timestamps:?}");
        assert_eq!(timestamps[0], (1000, 3000));
        assert_eq!(timestamps[1], (8000, 10000));
        assert_eq!(duration, 10000);
    }

    #[test]
    fn empty_subtitle_file() {
        let tmp = TempDir::new().unwrap();
        let path = write_file(&tmp, "empty.srt", "");
        let (timestamps, duration) = extract_timestamps(&path);
        assert!(timestamps.is_empty());
        assert_eq!(duration, 0);
    }

    #[test]
    fn srt_with_empty_text_lines_skipped() {
        let content =
            "1\n00:00:01,000 --> 00:00:03,000\nHello\n\n2\n00:00:05,000 --> 00:00:07,000\n\n";
        let tmp = TempDir::new().unwrap();
        let path = write_file(&tmp, "sparse.srt", content);
        let (timestamps, duration) = extract_timestamps(&path);
        assert_eq!(timestamps.len(), 1);
        assert_eq!(timestamps[0], (1000, 3000));
        assert_eq!(duration, 3000);
    }

    #[test]
    fn duration_is_max_end_time() {
        let content =
            "1\n00:00:01,000 --> 00:00:03,000\nFirst\n\n2\n00:00:20,000 --> 00:00:25,000\nLast\n";
        let tmp = TempDir::new().unwrap();
        let path = write_file(&tmp, "duration.srt", content);
        let (_, duration) = extract_timestamps(&path);
        assert_eq!(duration, 25000);
    }

    #[test]
    fn returns_sorted_timestamps() {
        let tmp = TempDir::new().unwrap();
        let path = write_file(&tmp, "ordered.srt", SRT_CONTENT);
        let (timestamps, _) = extract_timestamps(&path);
        let starts: Vec<i64> = timestamps.iter().map(|&(s, _)| s).collect();
        let mut sorted = starts.clone();
        sorted.sort_unstable();
        assert_eq!(starts, sorted);
    }

    // -----------------------------------------------------------------------
    // SubtitleValidator fixtures
    // -----------------------------------------------------------------------

    fn ref_lines() -> Vec<(i64, i64, &'static str)> {
        vec![
            (1000, 3000, "A"),
            (4000, 6000, "B"),
            (7000, 9000, "C"),
            (10000, 12000, "D"),
            (14000, 16000, "E"),
            (18000, 20000, "F"),
            (22000, 24000, "G"),
            (26000, 28000, "H"),
            (60000, 62000, "I"),
            (65000, 67000, "J"),
            (70000, 72000, "K"),
            (75000, 77000, "L"),
            (80000, 82000, "M"),
            (150000, 152000, "N"),
            (153000, 155000, "O"),
            (157000, 159000, "P"),
            (161000, 163000, "Q"),
            (165000, 167000, "R"),
            (170000, 172000, "S"),
            (175000, 177000, "T"),
            (240000, 242000, "U"),
            (250000, 252000, "V"),
            (260000, 262000, "W"),
        ]
    }

    fn matching_lines() -> Vec<(i64, i64, &'static str)> {
        vec![
            (1200, 3200, "A2"),
            (4200, 6200, "B2"),
            (7500, 9500, "C2"),
            (10500, 12500, "D2"),
            (14500, 16500, "E2"),
            (18500, 20500, "F2"),
            (22500, 24500, "G2"),
            (26500, 28500, "H2"),
            (60500, 62500, "I2"),
            (65500, 67500, "J2"),
            (70500, 72500, "K2"),
            (75500, 77500, "L2"),
            (80500, 82500, "M2"),
            (150500, 152500, "N2"),
            (153500, 155500, "O2"),
            (157500, 159500, "P2"),
            (161500, 163500, "Q2"),
            (165500, 167500, "R2"),
            (170500, 172500, "S2"),
            (175500, 177500, "T2"),
            (240500, 242500, "U2"),
            (250500, 252500, "V2"),
            (260500, 262500, "W2"),
        ]
    }

    fn mismatched_lines() -> Vec<(i64, i64, &'static str)> {
        vec![
            (5000, 7000, "X"),
            (20000, 22000, "Y"),
            (35000, 37000, "Z"),
            (50000, 52000, "AA"),
            (65000, 67000, "BB"),
            (80000, 82000, "CC"),
            (95000, 97000, "DD"),
            (110000, 112000, "EE"),
            (125000, 127000, "FF"),
            (140000, 142000, "GG"),
            (155000, 157000, "HH"),
            (170000, 172000, "II"),
            (185000, 187000, "JJ"),
            (200000, 202000, "KK"),
            (215000, 217000, "LL"),
            (230000, 232000, "MM"),
            (245000, 247000, "NN"),
            (260000, 262000, "OO"),
            (275000, 277000, "PP"),
            (290000, 292000, "QQ"),
        ]
    }

    fn make_match(subtitle_id: &str) -> SubtitleMatch {
        SubtitleMatch {
            language: "eng".to_string(),
            source: "test".to_string(),
            subtitle_id: subtitle_id.to_string(),
            release_name: "test-release".to_string(),
            format: "srt".to_string(),
            score: 0.8,
            hash_match: false,
        }
    }

    // -----------------------------------------------------------------------
    // TestSubtitleValidator
    // -----------------------------------------------------------------------

    #[test]
    fn validator_init_loads_reference() {
        let tmp = TempDir::new().unwrap();
        let ref_path = write_file(&tmp, "reference.srt", &make_srt(&ref_lines()));
        let validator = SubtitleValidator::new(&ref_path, 10000);
        assert!(!validator.ref_timestamps.is_empty());
        assert!(validator.ref_duration > 0);
    }

    #[test]
    fn score_candidate_matching_is_high() {
        let tmp = TempDir::new().unwrap();
        let ref_path = write_file(&tmp, "reference.srt", &make_srt(&ref_lines()));
        let cand_path = write_file(&tmp, "candidate.srt", &make_srt(&matching_lines()));
        let validator = SubtitleValidator::new(&ref_path, 10000);
        let score = validator.score_candidate(&cand_path).unwrap();
        assert!(score >= 0.7, "expected >= 0.7, got {score}");
    }

    #[test]
    fn score_candidate_mismatched_is_low() {
        let tmp = TempDir::new().unwrap();
        let ref_path = write_file(&tmp, "reference.srt", &make_srt(&ref_lines()));
        let cand_path = write_file(&tmp, "candidate.srt", &make_srt(&mismatched_lines()));
        let validator = SubtitleValidator::new(&ref_path, 10000);
        let score = validator.score_candidate(&cand_path).unwrap();
        assert!(score < 0.5, "expected < 0.5, got {score}");
    }

    #[test]
    fn validate_candidates_filters_by_threshold() {
        let tmp = TempDir::new().unwrap();
        let ref_path = write_file(&tmp, "reference.srt", &make_srt(&ref_lines()));
        let good_path = write_file(&tmp, "good.srt", &make_srt(&matching_lines()));
        let bad_path = write_file(&tmp, "bad.srt", &make_srt(&mismatched_lines()));

        let validator = SubtitleValidator::new(&ref_path, 10000);
        let candidates = vec![
            (make_match("good"), good_path),
            (make_match("bad"), bad_path),
        ];
        let results = validator.validate_candidates(&candidates, 0.5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.subtitle_id, "good");
        assert!(results[0].2 >= 0.5);
    }

    #[test]
    fn validate_candidates_sorted_by_score_descending() {
        let tmp = TempDir::new().unwrap();
        let ref_path = write_file(&tmp, "reference.srt", &make_srt(&ref_lines()));
        let cand1_path = write_file(&tmp, "cand1.srt", &make_srt(&matching_lines()));
        let cand2_path = write_file(&tmp, "cand2.srt", &make_srt(&mismatched_lines()));

        let validator = SubtitleValidator::new(&ref_path, 10000);
        let candidates = vec![
            (make_match("bad"), cand2_path),
            (make_match("good"), cand1_path),
        ];
        let results = validator.validate_candidates(&candidates, 0.0);
        assert_eq!(results.len(), 2);
        assert!(results[0].2 >= results[1].2);
        assert_eq!(results[0].0.subtitle_id, "good");
    }

    #[test]
    fn validate_candidates_empty_list() {
        let tmp = TempDir::new().unwrap();
        let ref_path = write_file(&tmp, "reference.srt", &make_srt(&ref_lines()));
        let validator = SubtitleValidator::new(&ref_path, 10000);
        let results = validator.validate_candidates(&[], 0.5);
        assert!(results.is_empty());
    }

    #[test]
    fn custom_window_size() {
        let tmp = TempDir::new().unwrap();
        let ref_path = write_file(&tmp, "reference.srt", &make_srt(&ref_lines()));
        let validator = SubtitleValidator::new(&ref_path, 5000);
        assert_eq!(validator.window_ms, 5000);
    }

    #[test]
    fn score_empty_candidate() {
        let tmp = TempDir::new().unwrap();
        let ref_path = write_file(&tmp, "reference.srt", &make_srt(&ref_lines()));
        let cand_path = write_file(&tmp, "candidate.srt", "");
        let validator = SubtitleValidator::new(&ref_path, 10000);
        let score = validator.score_candidate(&cand_path).unwrap();
        assert_eq!(score, 0.0);
    }

    #[test]
    fn score_candidate_missing_file_returns_err() {
        let tmp = TempDir::new().unwrap();
        let ref_path = write_file(&tmp, "reference.srt", &make_srt(&ref_lines()));
        let validator = SubtitleValidator::new(&ref_path, 10000);
        let missing = tmp.path().join("does_not_exist.srt");
        let result = validator.score_candidate(&missing);
        assert!(result.is_err(), "expected Err for unreadable candidate");
    }

    #[test]
    fn validate_candidates_skips_unreadable_candidate() {
        let tmp = TempDir::new().unwrap();
        let ref_path = write_file(&tmp, "reference.srt", &make_srt(&ref_lines()));
        let good_path = write_file(&tmp, "good.srt", &make_srt(&matching_lines()));
        let missing = tmp.path().join("missing.srt");

        let validator = SubtitleValidator::new(&ref_path, 10000);
        let candidates = vec![
            (make_match("missing"), missing),
            (make_match("good"), good_path),
        ];
        // The unreadable candidate is skipped via normal control flow (no panic).
        let results = validator.validate_candidates(&candidates, 0.5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.subtitle_id, "good");
    }

    // -----------------------------------------------------------------------
    // Non-dialogue prefix bug documentation test
    // -----------------------------------------------------------------------

    /// Test: subtitle file whose FIRST events are non-dialogue (signs/songs)
    /// followed by dialogue events.
    ///
    /// KNOWN BUG IN PYTHON (from project memory): the Python validator.py
    /// previously used a first-dialogue heuristic that failed when non-dialogue
    /// events appeared before the first dialogue event, causing 3/8 test files
    /// to fail validation.
    ///
    /// CURRENT STATE: the current Python source (`validator.py` as ported here)
    /// uses `classify_styles()` which iterates ALL events and collects per-style
    /// aggregate metrics before filtering.  This means leading non-dialogue events
    /// do NOT cause any issues — the style classifier sees them and correctly marks
    /// them as non-dialogue.  The bug is NOT present in the current Python source.
    ///
    /// The Rust port faithfully reproduces the CURRENT Python behaviour: leading
    /// non-dialogue events are handled correctly.
    #[test]
    fn leading_non_dialogue_events_do_not_skip_dialogue() {
        // First 3 events are positioned signs (non-dialogue), then 3 dialogue events.
        let content = "\
[Script Info]
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1
Style: Signs,Arial,36,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.50,0:00:05.00,Signs,,0,0,0,,{\\pos(100,200)}Sign text A
Dialogue: 0,0:00:01.00,0:00:06.00,Signs,,0,0,0,,{\\pos(100,200)}Sign text B
Dialogue: 0,0:00:02.00,0:00:07.00,Signs,,0,0,0,,{\\pos(100,200)}Sign text C
Dialogue: 0,0:00:10.00,0:00:12.00,Default,,0,0,0,,First dialogue line
Dialogue: 0,0:00:14.00,0:00:16.00,Default,,0,0,0,,Second dialogue line
Dialogue: 0,0:00:18.00,0:00:20.00,Default,,0,0,0,,Third dialogue line
";
        let tmp = TempDir::new().unwrap();
        let path = write_file(&tmp, "leading_signs.ass", content);
        let (timestamps, duration) = extract_timestamps(&path);

        // The 3 Default dialogue events should be included.
        assert_eq!(
            timestamps.len(),
            3,
            "expected 3 dialogue timestamps, got {}: {timestamps:?}",
            timestamps.len()
        );
        assert_eq!(timestamps[0], (10000, 12000));
        assert_eq!(timestamps[1], (14000, 16000));
        assert_eq!(timestamps[2], (18000, 20000));
        assert_eq!(duration, 20000);
    }
}
