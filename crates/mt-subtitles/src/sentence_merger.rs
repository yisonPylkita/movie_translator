//! Sentence-level grouping for seq2seq subtitle translation.
//!
//! Groups consecutive subtitle lines into complete sentences before
//! translation, then splits the translated output back to match original
//! line boundaries.  Independent sentences are separated with ` || `
//! (double-pipe), which the target models pass through at 93%+ fidelity.
//!
//! Key rules (experimentally verified):
//! - Lines without terminal punctuation (.!?) are fragments -> merge with next
//! - Ellipsis (`...`) is a continuation marker -> merge
//! - Speaker dash lines (`- text`) are NEVER merged — model drops one speaker
//! - The `||` separator preserves sentence boundaries through translation
//! - `||` groups are capped at [`MAX_BATCH_SENTENCES`] (3)
//! - Pipe characters are stripped from output to prevent leaking
//! - Proportional word-count splitting redistributes merged output

use regex::Regex;

/// Maximum sentences in a single `||`-separated group.
/// Beyond this the BiDi model loses separators and enters repetition loops.
const MAX_BATCH_SENTENCES: usize = 3;

/// Maximum total words in a single `||`-separated group.
const MAX_BATCH_WORDS: usize = 60;

/// Lines with this many words or fewer get their own solo group.
const SHORT_LINE_MAX_WORDS: usize = 3;

lazy_static::lazy_static! {
    /// Terminal punctuation: . ! ? optionally followed by closing quotes/parens.
    static ref TERMINAL_RE: Regex = Regex::new(r#"[.!?]["\')»\]"]*\s*$"#).unwrap();

    /// Ellipsis at end of line — continuation, NOT terminal.
    static ref ELLIPSIS_RE: Regex = Regex::new(r#"\.{2,}["\')»\]"]*\s*$"#).unwrap();

    /// Speaker dash at start of line.
    static ref SPEAKER_DASH_RE: Regex = Regex::new(r"^[\-\u2014\u2013]\s*\S").unwrap();

    /// Leading ellipsis used as continuation marker.
    static ref LEADING_ELLIPSIS_RE: Regex = Regex::new(r"^\.{2,}\s*").unwrap();

    /// Trailing ellipsis.
    static ref TRAILING_ELLIPSIS_RE: Regex = Regex::new(r"\s*\.{2,}$").unwrap();

    /// Pipe characters that may leak from `||` separator.
    static ref PIPE_RE: Regex = Regex::new(r"\s*\|+\s*").unwrap();
}

/// A group of consecutive subtitle lines to be translated together.
#[derive(Debug, Clone)]
pub struct TranslationGroup {
    /// Indices into the original texts list.
    pub line_indices: Vec<usize>,
    /// True when lines were merged because they form fragments of one sentence
    /// (space-joined). False when lines are independent complete sentences
    /// (joined with `||`).
    pub is_fragment_merge: bool,
}

impl TranslationGroup {
    pub fn new_solo(index: usize) -> Self {
        TranslationGroup {
            line_indices: vec![index],
            is_fragment_merge: false,
        }
    }

    pub fn new_fragment(index: usize) -> Self {
        TranslationGroup {
            line_indices: vec![index],
            is_fragment_merge: true,
        }
    }
}

/// Return `true` if `text` ends a complete sentence.
///
/// Terminal: . ! ? (optionally followed by closing quotes/parens).
/// Non-terminal: comma, colon, semicolon, ellipsis, no punctuation.
pub fn is_sentence_end(text: &str) -> bool {
    let stripped = text.trim();
    if stripped.is_empty() {
        return false;
    }
    // Ellipsis is explicitly non-terminal (continuation)
    if ELLIPSIS_RE.is_match(stripped) {
        return false;
    }
    TERMINAL_RE.is_match(stripped)
}

/// Return `true` if `text` starts with a dialogue dash (-, —, –).
pub fn is_speaker_line(text: &str) -> bool {
    let stripped = text.trim();
    if stripped.is_empty() {
        return false;
    }
    SPEAKER_DASH_RE.is_match(stripped)
}

/// Group consecutive subtitle lines into translation units.
pub fn group_lines(texts: &[String]) -> Vec<TranslationGroup> {
    if texts.is_empty() {
        return vec![];
    }

    let n = texts.len();
    let mut i = 0;
    let mut groups: Vec<TranslationGroup> = Vec::new();

    while i < n {
        // Speaker lines are always solo
        if is_speaker_line(&texts[i]) {
            groups.push(TranslationGroup::new_solo(i));
            i += 1;
            continue;
        }

        // Check if current line is a fragment (not a sentence end)
        if !is_sentence_end(&texts[i]) {
            // Start a fragment merge group
            let mut group = TranslationGroup::new_fragment(i);
            i += 1;
            // Keep merging until we hit a sentence end or run out of lines
            while i < n {
                if is_speaker_line(&texts[i]) {
                    break;
                }
                // If the previous line ended with ellipsis and this line
                // starts with a capital letter, it's a new sentence despite
                // the ellipsis — stop merging before it.
                let prev_text = texts[group.line_indices[group.line_indices.len() - 1]].trim();
                let curr_text = texts[i].trim();
                if ELLIPSIS_RE.is_match(prev_text)
                    && !curr_text.is_empty()
                    && curr_text.chars().next().unwrap().is_uppercase()
                {
                    break;
                }
                group.line_indices.push(i);
                if is_sentence_end(&texts[i]) {
                    i += 1;
                    break;
                }
                i += 1;
            }
            groups.push(group);
            continue;
        }

        // Very short complete sentences get their own group
        if texts[i].split_whitespace().count() <= SHORT_LINE_MAX_WORDS {
            groups.push(TranslationGroup::new_solo(i));
            i += 1;
            continue;
        }

        // Batch consecutive complete sentences, capped at MAX_BATCH_SENTENCES
        let mut group = TranslationGroup::new_solo(i);
        let mut current_words = texts[i].split_whitespace().count();
        i += 1;
        while i < n && group.line_indices.len() < MAX_BATCH_SENTENCES {
            if is_speaker_line(&texts[i]) {
                break;
            }
            if !is_sentence_end(&texts[i]) {
                break;
            }
            let next_words = texts[i].split_whitespace().count();
            if current_words + next_words > MAX_BATCH_WORDS {
                break;
            }
            if next_words <= SHORT_LINE_MAX_WORDS {
                break;
            }
            group.line_indices.push(i);
            current_words += next_words;
            i += 1;
        }
        groups.push(group);
    }

    groups
}

/// Build translation input string for a single group.
///
/// Fragment-merged lines are space-joined.  Independent sentences are
/// joined with ` || `.
pub fn build_input(texts: &[String], group: &TranslationGroup) -> String {
    let lines: Vec<&str> = group
        .line_indices
        .iter()
        .map(|&idx| texts[idx].as_str())
        .collect();
    if group.is_fragment_merge {
        let mut cleaned = Vec::new();
        for (i, &line) in lines.iter().enumerate() {
            let mut text = line.to_string();
            // Strip trailing ellipsis if the next line starts with one
            if i < lines.len() - 1 {
                let next_line = lines[i + 1].trim();
                if TRAILING_ELLIPSIS_RE.is_match(&text) && next_line.starts_with('.') {
                    text = TRAILING_ELLIPSIS_RE.replace(&text, "").to_string();
                }
            }
            // Strip leading ellipsis if the previous line ended with one
            if i > 0 {
                let prev_line = lines[i - 1].trim();
                if LEADING_ELLIPSIS_RE.is_match(&text) && prev_line.ends_with('.') {
                    text = LEADING_ELLIPSIS_RE.replace(&text, "").to_string();
                }
            }
            cleaned.push(text.trim().to_string());
        }
        cleaned.join(" ")
    } else {
        lines.join(" || ")
    }
}

/// Remove orphaned `|` and `||` tokens from text.
pub fn strip_pipes(text: &str) -> String {
    let cleaned = PIPE_RE.replace_all(text, " ");
    cleaned.trim().to_string()
}

/// Split translated text back to the original line count for `group`.
pub fn split_output(
    translated: &str,
    group: &TranslationGroup,
    original_texts: &[String],
) -> Vec<String> {
    let n_lines = group.line_indices.len();
    if n_lines == 1 {
        return vec![strip_pipes(translated.trim())];
    }

    if !group.is_fragment_merge {
        // Split on `||`
        let parts: Vec<&str> = translated.split("||").collect();
        if parts.len() == n_lines {
            return parts.iter().map(|p| strip_pipes(p.trim())).collect();
        }
        // Fallback: proportional word-count split
        return proportional_split(translated, group, original_texts);
    }

    proportional_split(translated, group, original_texts)
}

/// Redistribute translated words proportionally to original word counts.
fn proportional_split(
    translated: &str,
    group: &TranslationGroup,
    original_texts: &[String],
) -> Vec<String> {
    let cleaned = strip_pipes(translated);
    let words: Vec<&str> = cleaned.split_whitespace().collect();
    let total_translated = words.len();
    if total_translated == 0 {
        return vec!["".to_string(); group.line_indices.len()];
    }

    let orig_counts: Vec<usize> = group
        .line_indices
        .iter()
        .map(|&idx| original_texts[idx].split_whitespace().count().max(1))
        .collect();
    let total_orig: usize = orig_counts.iter().sum();

    let mut result = Vec::new();
    let mut used = 0;
    let mut remaining_segments = orig_counts.len();

    for &count in orig_counts.iter() {
        remaining_segments -= 1;
        if remaining_segments == 0 {
            result.push(words[used..].join(" "));
        } else {
            let remaining_words = total_translated - used;
            let mut share =
                (count as f64 / total_orig as f64 * total_translated as f64).round() as usize;
            share = share.max(1.min(remaining_words));
            share = share.min(remaining_words.saturating_sub(remaining_segments));
            result.push(words[used..used + share].join(" "));
            used += share;
        }
    }

    result
}

/// Group texts and build merged translation inputs.
///
/// Returns `(merged_texts, groups)` where each element of `merged_texts`
/// corresponds to one `TranslationGroup`.
pub fn merge_for_translation(texts: &[String]) -> (Vec<String>, Vec<TranslationGroup>) {
    let groups = group_lines(texts);
    let merged: Vec<String> = groups.iter().map(|g| build_input(texts, g)).collect();
    (merged, groups)
}

/// Split translated texts back to match the original line count.
///
/// Returns a flat list aligned 1-to-1 with `original_texts`.
pub fn unmerge_translations(
    translated_texts: &[String],
    groups: &[TranslationGroup],
    original_texts: &[String],
) -> Vec<String> {
    let mut result = vec!["".to_string(); original_texts.len()];
    for (translated, group) in translated_texts.iter().zip(groups.iter()) {
        let parts = split_output(translated, group, original_texts);
        for (&idx, part) in group.line_indices.iter().zip(parts.iter()) {
            result[idx] = part.clone();
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentence_end() {
        assert!(is_sentence_end("Hello."));
        assert!(is_sentence_end("Hello!"));
        assert!(is_sentence_end("Hello?"));
        assert!(is_sentence_end("\"Hello.\""));
        assert!(!is_sentence_end("Hello,"));
        assert!(!is_sentence_end("Hello..."));
        assert!(!is_sentence_end("Hello"));
        assert!(!is_sentence_end(""));
        assert!(!is_sentence_end("  "));
    }

    #[test]
    fn test_speaker_line() {
        assert!(is_speaker_line("- Hello"));
        assert!(is_speaker_line("— Hello"));
        assert!(is_speaker_line("– Hello"));
        assert!(!is_speaker_line("Hello"));
        assert!(!is_speaker_line(""));
    }

    #[test]
    fn test_group_lines_short_sentences_solo() {
        // "Hello." (1 word) and "How are you?" (3 words) are both
        // <= SHORT_LINE_MAX_WORDS (3), so each gets its own group.
        let texts: Vec<String> = vec!["Hello.".into(), "How are you?".into()];
        let groups = group_lines(&texts);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].line_indices, vec![0]);
        assert_eq!(groups[1].line_indices, vec![1]);
    }

    #[test]
    fn test_group_lines_longer_sentences_batch() {
        // Longer (>3 words) complete sentences batch together.
        let texts: Vec<String> = vec![
            "I am going to the store.".into(),
            "Do you need anything?".into(),
        ];
        let groups = group_lines(&texts);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].line_indices, vec![0, 1]);
        assert!(!groups[0].is_fragment_merge);
    }

    #[test]
    fn test_group_lines_fragment() {
        let texts: Vec<String> = vec!["In the beginning,".into(), "there was nothing.".into()];
        let groups = group_lines(&texts);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].line_indices, vec![0, 1]);
        assert!(groups[0].is_fragment_merge);
    }

    #[test]
    fn test_group_lines_speaker_solo() {
        let texts: Vec<String> = vec!["- Hello.".into(), "- How are you?".into()];
        let groups = group_lines(&texts);
        assert_eq!(groups.len(), 2);
    }

    #[test]
    fn test_build_input_fragment() {
        let texts: Vec<String> = vec!["In the beginning,".into(), "there was nothing.".into()];
        let group = TranslationGroup {
            line_indices: vec![0, 1],
            is_fragment_merge: true,
        };
        let input = build_input(&texts, &group);
        assert_eq!(input, "In the beginning, there was nothing.");
    }

    #[test]
    fn test_build_input_separator() {
        let texts: Vec<String> = vec!["Hello.".into(), "How are you?".into()];
        let group = TranslationGroup {
            line_indices: vec![0, 1],
            is_fragment_merge: false,
        };
        let input = build_input(&texts, &group);
        assert_eq!(input, "Hello. || How are you?");
    }

    #[test]
    fn test_round_trip() {
        // Use long enough (>3 words) sentences so they batch together
        let texts: Vec<String> = vec![
            "I am going to the store.".into(),
            "Do you need anything from there?".into(),
            "Please pick up some milk.".into(),
        ];
        let (merged, groups) = merge_for_translation(&texts);
        assert!(merged.len() <= 2);

        // Simulate translation (identity)
        let unmerged = unmerge_translations(&merged, &groups, &texts);
        assert_eq!(unmerged.len(), 3);
    }

    #[test]
    fn test_short_sentences_round_trip() {
        // Short sentences each get their own group
        let texts: Vec<String> = vec!["Hi.".into(), "Bye.".into()];
        let (merged, groups) = merge_for_translation(&texts);
        assert_eq!(merged.len(), 2);
        assert_eq!(groups.len(), 2);

        let unmerged = unmerge_translations(&merged, &groups, &texts);
        assert_eq!(unmerged, vec!["Hi.", "Bye."]);
    }

    #[test]
    fn test_strip_pipes() {
        // " || " -> "   " (the regex \s*|+\s* captures surrounding spaces)
        assert_eq!(strip_pipes("hello || world"), "hello world");
        assert_eq!(strip_pipes("no pipes"), "no pipes");
    }
}
