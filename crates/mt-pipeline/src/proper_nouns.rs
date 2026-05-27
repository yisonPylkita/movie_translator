//! Proper-noun detection for translation protection.
//!
//! Faithful port of `movie_translator/translation/proper_nouns.py`. Builds a set
//! of character names / proper nouns that should not be translated, using
//! capitalization heuristics over the English subtitle dialogue.
//!
//! Every regex pattern, capitalization heuristic, stopword/filter list and the
//! `count >= 3` threshold mirror the Python implementation exactly.

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use regex::Regex;

// Tokens used to detect lowercase occurrences in the corpus.
static TOKEN_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[a-zA-Z']+").unwrap());
// Words after an honorific (boosted).
static HONORIFIC_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?:Sir|Lord|Lady|Princess|Prince|King|Queen|Master|Miss)\s+([A-Z][a-z]+)")
        .unwrap()
});
// Capitalized word in direct address (followed by comma, ! or ?).
static DIRECT_ADDRESS_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"([A-Z][a-z]{2,})[,!?]").unwrap());
// A whole (stripped) line that is just a single Title-case word.
static STANDALONE_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[A-Z][a-z]+$").unwrap());
// A mid-sentence Title-case word (>= 3 chars).
static MIDSENTENCE_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[A-Z][a-z]{2,}$").unwrap());

const HONORIFICS: &[&str] = &[
    "sir", "lord", "lady", "princess", "prince", "king", "queen", "master", "miss",
];

/// Common English words that are capitalized mid-sentence in subtitles but are
/// NOT proper nouns. Mirrors `common_false_positives` in the Python source.
const COMMON_FALSE_POSITIVES: &[&str] = &[
    // Articles, conjunctions, pronouns
    "The", "This", "That", "These", "Those", "What", "When", "Where", "Which", "Who", "How", "Why",
    "But", "And", "Yet", "For", "Nor", "Not", "Now", "Yes", "Yeah", "You", "Your", "Their", "Such",
    "Some", "Each", "Every", // Greetings + interjections
    "Hey", "Hi", "Hello", "Heya", "Oh", "Ah", "Eh", "Huh", "Wow", "Whoa", "Yo", "Goodbye", "Bye",
    "Welcome", // Discourse markers
    "Well", "So", "Also", "Anyway", "Anyhow", "Indeed", "However", "Besides", "Maybe", "Perhaps",
    "Supposedly", "Obviously", "Apparently", "Honestly", "Right", "Wrong", "True", "False", "Fine",
    "Sure", // Commands / direct address
    "Look", "Come", "Go", "Stop", "Wait", "Listen", "Hear", "See", "Run", "Hurry", "Move", "Stay",
    "Stand", "Sit", "Sleep", "Help", "Quiet", "Silence", "Enough", "Begin", "Start", "Finish",
    "Continue", "Return", "Forward", "Back", "Onward", "Charge", "Fire", "Attack", "Defend",
    // Adjectives
    "Big", "Small", "Old", "Young", "New", "Good", "Bad", "Great", "Little", "Strong", "Weak",
    "Brave", "Quick", "Slow", "Long", "Short", "Best", "Worst", "Hard", "Easy", "Tough", "Cool",
    "Hot", "Cold", "Crazy", // Adverbs / qualifiers
    "Never", "Always", "Often", "Sometimes", "Quickly", "Slowly", "Finally", "Suddenly", "Already",
    "Just", "Even", "Still", "Only", "Almost", "Alright", "Okay", "Damn", "Hell", "Heaven",
    // Politeness / fillers
    "Please", "Sorry", "Pardon", "Excuse", "Thanks", "Thank",
    // Common nouns sometimes capitalised at line start
    "God", "Idiot", "Fool", "Bastard", "Stupid", "Loser", "Coward", "Women", "Men", "Boy", "Girl",
    "Man", "Woman", "Boys", "Girls", "Brother", "Sister", "Bro", "Sis", "Mother", "Father", "Mom",
    "Dad", "Friend", "Friends", "Family", "Enemy", "Enemies", "Era", "Past", "Future", "Branch",
    "Empire", "Pirate", "Pirates", "Ship", "Sea", "Ocean", "Sky", "Day", "Night", "World",
    // Verbs that frequently start exclamations
    "Get", "Let", "Make", "Take", "Give", "Bring", "Tell", "Show", "Try", "Keep", "Open", "Close",
    "Find", "Lose", "Win", "Eat", "Drink", "Speak", "Talk", "Shout", "Yell", "Scream", "Cry",
    "Laugh", "Smile", "Fight", "Kill", "Die", "Live", "Love", "Hate", "Believe", "Trust", "Forget",
    "Remember", "Understood", "Gathered", "Canceled", "Cancelled", "Being",
];

/// Capitalize the first ASCII letter of `s`, lowercasing the rest — mirrors
/// Python's `str.capitalize()` for the all-ASCII honorific words.
fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => {
            first.to_ascii_uppercase().to_string() + &chars.as_str().to_ascii_lowercase()
        }
    }
}

/// Python `str.strip(chars)`: trim leading/trailing chars from the given set.
fn strip_chars(s: &str, chars: &[char]) -> String {
    s.trim_matches(|c| chars.contains(&c)).to_string()
}

/// Extract likely proper nouns from English subtitle text.
///
/// Faithful port of `extract_proper_nouns_from_subtitles`.
pub fn extract_proper_nouns_from_subtitles(dialogue_texts: &[String]) -> HashSet<String> {
    // Track lowercase forms anywhere in the corpus — if a token appears
    // lowercase, it's almost certainly an English word, not a proper noun.
    let mut seen_lowercase: HashSet<String> = HashSet::new();
    for text in dialogue_texts {
        for m in TOKEN_RE.find_iter(text) {
            let tok = m.as_str();
            if tok == tok.to_lowercase() {
                seen_lowercase.insert(tok.to_string());
            }
        }
    }

    let mut cap_word_counts: HashMap<String, i64> = HashMap::new();
    fn bump(counts: &mut HashMap<String, i64>, word: &str, by: i64) {
        *counts.entry(word.to_string()).or_insert(0) += by;
    }

    for text in dialogue_texts {
        // Words after honorifics (boost).
        for caps in HONORIFIC_RE.captures_iter(text) {
            bump(&mut cap_word_counts, &caps[1], 5);
        }

        // Capitalized words in direct address (followed by comma, ! or ?).
        for caps in DIRECT_ADDRESS_RE.captures_iter(text) {
            bump(&mut cap_word_counts, &caps[1], 2);
        }

        // Standalone exclamation (whole line is just a name).
        let stripped = strip_chars(text.trim(), &['!', '?', '.']);
        let stripped = stripped.trim();
        if STANDALONE_RE.is_match(stripped) {
            bump(&mut cap_word_counts, stripped, 3);
        }

        // Count all mid-sentence capitalized words.
        for (i, word) in text.split_whitespace().enumerate() {
            if i == 0 {
                continue; // skip sentence start
            }
            let clean = strip_chars(word, &['.', ',', '!', '?', ';', ':', '"', '\'', '-']);
            if MIDSENTENCE_RE.is_match(&clean) {
                bump(&mut cap_word_counts, &clean, 1);
            }
        }
    }

    // Build the false-positive set (list + capitalized honorifics).
    let mut common_false_positives: HashSet<String> =
        COMMON_FALSE_POSITIVES.iter().map(|s| s.to_string()).collect();
    for h in HONORIFICS {
        common_false_positives.insert(capitalize(h));
    }

    let mut names: HashSet<String> = HashSet::new();
    for (word, count) in &cap_word_counts {
        if *count < 3 {
            continue;
        }
        if common_false_positives.contains(word) {
            continue;
        }
        // If we've also seen the word lowercased, it's English, not a name.
        if seen_lowercase.contains(&word.to_lowercase()) {
            continue;
        }
        names.insert(word.clone());
    }

    if !names.is_empty() {
        let mut sorted: Vec<&String> = names.iter().collect();
        sorted.sort();
        tracing::info!("Detected proper nouns for translation protection: {sorted:?}");
    }

    names
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(lines: &[&str]) -> Vec<String> {
        let owned: Vec<String> = lines.iter().map(|s| s.to_string()).collect();
        let mut v: Vec<String> = extract_proper_nouns_from_subtitles(&owned)
            .into_iter()
            .collect();
        v.sort();
        v
    }

    // Expected sets generated by running the Python
    // `extract_proper_nouns_from_subtitles` on these exact inputs via
    // `uv run python` (see commit message / task notes).

    #[test]
    fn honorific_and_direct_address() {
        // "Griffith" gets honorific boost (+5) plus direct-address/mid-sentence
        // hits; "Guts" hits direct address repeatedly.
        let lines = [
            "Sir Griffith, the army awaits.",
            "Guts! Where are you going?",
            "I trust Griffith with my life.",
            "Guts, come back here!",
            "We follow Griffith into battle.",
            "Guts is the strongest of us.",
        ];
        assert_eq!(run(&lines), vec!["Griffith", "Guts"]);
    }

    #[test]
    fn lowercase_occurrence_filters_english_word() {
        // "Stop" appears capitalized 3+ times but also lowercase → dropped.
        // (It's also in the false-positive list, doubly excluded.)
        let lines = [
            "Stop! You can't do this.",
            "Stop, I said!",
            "Please Stop now.",
            "I want you to stop running.",
        ];
        assert_eq!(run(&lines), Vec::<String>::new());
    }

    #[test]
    fn standalone_name_lines() {
        // A bare name line counts +3, enough on its own.
        let lines = ["Casca!", "Casca?", "Where is Casca going now?"];
        assert_eq!(run(&lines), vec!["Casca"]);
    }

    #[test]
    fn below_threshold_dropped() {
        // Single mid-sentence hit (+1) is below the count>=3 threshold.
        let lines = ["We sailed past Romsdal yesterday."];
        assert_eq!(run(&lines), Vec::<String>::new());
    }

    #[test]
    fn empty_input() {
        assert_eq!(run(&[]), Vec::<String>::new());
    }
}
