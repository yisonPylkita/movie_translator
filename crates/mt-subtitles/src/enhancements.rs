//! Translation enhancements: phrase lookup, placeholder protection,
//! preprocessing, postprocessing, and fallback logic.
//!
//! Shared by every translation backend (Apple, MLX) to improve output
//! quality: protect proper nouns and numbers from translation, handle
//! common phrases idiomatically, and clean up known bad translations.

use std::collections::{HashMap, HashSet};

use regex::{Captures, Regex, escape as regex_escape};
use tracing::warn;

lazy_static::lazy_static! {
    // ── Single-word phrase map ──
    static ref PHRASE_BASE_MAP: HashMap<&'static str, &'static str> = {
        let mut m = HashMap::new();
        m.insert("yes", "tak");
        m.insert("no", "nie");
        m.insert("maybe", "może");
        m.insert("wait", "czekaj");
        m.insert("stop", "stop");
        m.insert("go", "idź");
        m.insert("help", "pomocy");
        m.insert("please", "proszę");
        m.insert("thanks", "dzięki");
        m.insert("sorry", "przepraszam");
        m.insert("okay", "dobrze");
        m.insert("ok", "ok");
        m.insert("fine", "dobrze");
        m.insert("sure", "jasne");
        m.insert("never", "nigdy");
        m.insert("always", "zawsze");
        m.insert("hello", "cześć");
        m.insert("hi", "cześć");
        m.insert("bye", "pa");
        m.insert("goodbye", "do widzenia");
        m.insert("huh", "co");
        m.insert("what", "co");
        m.insert("right", "racja");
        m.insert("really", "naprawdę");
        m.insert("seriously", "poważnie");
        m.insert("exactly", "dokładnie");
        m.insert("understood", "zrozumiałem");
        m.insert("impossible", "niemożliwe");
        m.insert("damn", "cholera");
        m.insert("dunno", "nie wiem");
        m.insert("yo", "hej");
        m.insert("hey", "hej");
        m.insert("listen", "słuchaj");
        m.insert("look", "patrz");
        m.insert("run", "uciekaj");
        m.insert("enough", "wystarczy");
        m.insert("idiot", "idioto");
        m.insert("liar", "kłamca");
        m.insert("unbelievable", "niewiarygodne");
        m.insert("march", "naprzód");
        m.insert("charge", "do ataku");
        m
    };

    // ── Multi-word phrase map ──
    static ref MULTI_WORD_PHRASES: HashMap<&'static str, &'static str> = {
        let mut m = HashMap::new();
        m.insert("thank you", "dziękuję");
        m.insert("i see", "rozumiem");
        m.insert("i know", "wiem");
        m.insert("of course", "oczywiście");
        m.insert("excuse me", "przepraszam");
        m.insert("good luck", "powodzenia");
        m.insert("come on", "no dalej");
        m.insert("no way", "nie ma mowy");
        m.insert("damn it", "cholera");
        m.insert("what the hell", "co do cholery");
        m.insert("got it", "rozumiem");
        m.insert("hold on", "chwileczkę");
        m.insert("not at all", "wcale nie");
        m.insert("no doubt", "bez dwóch zdań");
        m.insert("give me a break", "daj spokój");
        m.insert("shut up", "zamknij się");
        m.insert("how dare you", "jak śmiesz");
        m.insert("calm down", "spokojnie");
        m.insert("let me go", "puść mnie");
        m.insert("you idiot", "ty idioto");
        m.insert("are you okay", "wszystko dobrze");
        m.insert("good morning", "dzień dobry");
        m.insert("good night", "dobranoc");
        m.insert("way to go", "tak trzymaj");
        m.insert("get it together", "weź się w garść");
        m.insert("son of a bitch", "sukinsyn");
        m
    };

    // ── Idiom patterns ──
    static ref IDIOM_PATTERNS: Vec<(Regex, &'static str)> = vec![
        (Regex::new(r"\bbreak a leg\b").unwrap(), "good luck"),
        (Regex::new(r"\braining cats and dogs\b").unwrap(), "raining heavily"),
        (Regex::new(r"\bpiece of cake\b").unwrap(), "very easy"),
        (Regex::new(r"\bhit the nail on the head\b").unwrap(), "exactly right"),
        (Regex::new(r"\blet the cat out of the bag\b").unwrap(), "reveal a secret"),
        (Regex::new(r"\bonce in a blue moon\b").unwrap(), "very rarely"),
        (Regex::new(r"\bunder the weather\b").unwrap(), "feeling sick"),
        (Regex::new(r"\bspill the beans\b").unwrap(), "reveal a secret"),
        (Regex::new(r"\bbarking up the wrong tree\b").unwrap(), "looking in the wrong place"),
        (Regex::new(r"\bcost an arm and a leg\b").unwrap(), "very expensive"),
    ];

    // ── Post-translation fixes ──
    static ref POST_TRANSLATION_FIXES: Vec<(Regex, &'static str)> = vec![
        (Regex::new(r"\bMarzec\b").unwrap(), "Naprzód"),
        (Regex::new(r"\bDobra droga\b").unwrap(), "Tak trzymaj"),
        (Regex::new(r"^Ładuj!$").unwrap(), "Do ataku!"),
        (Regex::new(r"\bstrajk\b").unwrap(), "cios"),
        (Regex::new(r"Weź to razem").unwrap(), "Weź się w garść"),
    ];

    /// A line that is nothing but a placeholder tag + punctuation.
    pub static ref PLACEHOLDER_ONLY_RE: Regex = Regex::new(r"^__\w+__[.!?,;:…\s]*$").unwrap();
}

/// Normalization result for phrase lookup.
struct Normalized {
    base: String,
    punct: String,
    cap_pattern: Capitalization,
}

enum Capitalization {
    Upper,
    Title,
    Lower,
}

fn normalize_for_lookup(text: &str) -> Normalized {
    let stripped = text.trim();
    // Extract trailing punctuation
    let re = Regex::new(r"^(.*?)([.!?,;:…]+)?$").unwrap();
    let caps = re.captures(stripped).unwrap();
    let base = caps.get(1).map(|m| m.as_str().trim()).unwrap_or(stripped);
    let punct = caps.get(2).map(|m| m.as_str()).unwrap_or("");

    let cap_pattern = if base.chars().all(|c| c.is_uppercase() || !c.is_alphabetic()) {
        Capitalization::Upper
    } else if base
        .chars()
        .next()
        .map(|c| c.is_uppercase())
        .unwrap_or(false)
    {
        Capitalization::Title
    } else {
        Capitalization::Lower
    };

    Normalized {
        base: base.to_lowercase(),
        punct: punct.to_string(),
        cap_pattern,
    }
}

fn apply_formatting(translated: &str, punct: &str, cap_pattern: &Capitalization) -> String {
    let mut result = match cap_pattern {
        Capitalization::Upper => translated.to_uppercase(),
        Capitalization::Title => {
            if let Some(c) = translated.chars().next() {
                c.to_uppercase().to_string() + &translated[c.len_utf8()..]
            } else {
                translated.to_string()
            }
        }
        Capitalization::Lower => translated.to_string(),
    };
    result.push_str(punct);
    result
}

/// Preprocess a single text for translation.
///
/// Returns `(processed_text, was_mapped)` where `was_mapped` is `true` if
/// the text was handled by a phrase lookup (bypassing the model).
pub fn preprocess_for_translation(text: &str) -> (String, bool) {
    let norm = normalize_for_lookup(text);
    let base = norm.base.as_str();

    // Check single-word phrases
    if let Some(&translation) = PHRASE_BASE_MAP.get(base) {
        return (
            apply_formatting(translation, &norm.punct, &norm.cap_pattern),
            true,
        );
    }

    // Check multi-word phrases
    if let Some(&translation) = MULTI_WORD_PHRASES.get(base) {
        return (
            apply_formatting(translation, &norm.punct, &norm.cap_pattern),
            true,
        );
    }

    // Apply idiom patterns
    let mut processed = text.to_string();
    for (pattern, replacement) in IDIOM_PATTERNS.iter() {
        processed = pattern.replace_all(&processed, *replacement).to_string();
    }

    (processed, false)
}

/// Post-process a translated text: fix known bad translations, clean up.
pub fn postprocess_translation(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }

    let mut cleaned = text.trim().to_string();

    // Apply known-bad-translation fixes
    for (pattern, replacement) in POST_TRANSLATION_FIXES.iter() {
        cleaned = pattern.replace_all(&cleaned, *replacement).to_string();
    }

    cleaned = remove_dialogue_markers(&cleaned);
    cleaned = remove_repetition(&cleaned);
    cleaned = normalize_punctuation(&cleaned);

    cleaned
}

fn remove_dialogue_markers(text: &str) -> String {
    let trimmed = text.trim();

    // Try exclamation mark pattern: "- x! - x!"
    if let Some(rest) = trimmed.strip_prefix("- ")
        && let Some((first, second_rest)) = rest.split_once('!')
    {
        let second_rest = second_rest.trim();
        if let Some(second) = second_rest.strip_prefix("- ")
            && let Some(second_text) = second.strip_suffix('!').map(|s| s.trim())
            && !second_text.is_empty()
            && first.trim().to_lowercase() == second_text.to_lowercase()
        {
            return format!("{}!", first.trim());
        }
    }

    // Try punctuation pattern: "- x. - x." etc.
    for punct in &[".", "!", "?"] {
        if let Some(rest) = trimmed.strip_prefix("- ")
            && let Some((first, rest_after_punct)) = rest.split_once(punct)
        {
            let rest_after_punct = rest_after_punct.trim();
            if let Some(second_with_trailing) = rest_after_punct.strip_prefix("- ") {
                // The second part should end with the same punctuation
                if let Some(second) = second_with_trailing.strip_suffix(punct).map(|s| s.trim())
                    && first.trim().to_lowercase() == second.to_lowercase()
                {
                    return format!("{}{}", first.trim(), punct);
                }
            }
        }
    }

    text.to_string()
}

fn remove_repetition(text: &str) -> String {
    // Pattern: "word, word!" or "word, word."
    // Check for comma-separated repeated word before punctuation.
    let trimmed = text.trim();
    if let Some(comma_pos) = trimmed.find(',') {
        let first = trimmed[..comma_pos].trim();
        let after_comma = trimmed[comma_pos + 1..].trim();
        // Second part should be the same word followed by one punctuation char
        if after_comma.len() > 1 {
            let second_word = &after_comma[..after_comma.len() - 1];
            let last_char = &after_comma[after_comma.len() - 1..];
            if !first.is_empty()
                && first.to_lowercase() == second_word.to_lowercase()
                && matches!(last_char, "." | "!" | "?")
            {
                return format!("{first}{last_char}");
            }
        }
    }
    text.to_string()
}

fn normalize_punctuation(text: &str) -> String {
    // Collapse repeated punctuation: "!!" -> "!", "??" -> "?"
    let re = Regex::new(r"[.!?]{2,}").unwrap();
    let text = re
        .replace_all(text, |caps: &Captures| {
            let m = caps.get(0).unwrap().as_str();
            m.chars().next().unwrap().to_string()
        })
        .to_string();
    // Remove space before punctuation: "hello !" -> "hello!"
    let re = Regex::new(r"\s+([.!?])").unwrap();
    re.replace_all(&text, r"$1").to_string()
}

/// Patterns for content that should pass through translation untouched.
/// Use short non-English abbreviations to prevent the model from translating tags.
struct PlaceholderPattern {
    pattern: Regex,
    tag: &'static str,
}

fn placeholder_patterns() -> Vec<PlaceholderPattern> {
    vec![
        PlaceholderPattern {
            pattern: Regex::new(r"\b\d{1,3}([-.)\s]\d{2,4}){2,}\b").unwrap(),
            tag: "PH",
        }, // phone
        PlaceholderPattern {
            pattern: Regex::new(r"\b\d{1,2}[/:]\d{2}(?:[/:]\d{2,4})?\b").unwrap(),
            tag: "TM",
        }, // time/date
        PlaceholderPattern {
            pattern: Regex::new(r"https?://\S+").unwrap(),
            tag: "UR",
        }, // URL
        PlaceholderPattern {
            pattern: Regex::new(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b").unwrap(),
            tag: "NM",
        }, // Title Case names
    ]
}

/// Replace known proper nouns, numbers, URLs and Title Case names with
/// placeholders so the translation model doesn't mangle them.
///
/// Returns `(modified_text, placeholder_mapping)`.
pub fn extract_placeholders(
    text: &str,
    proper_nouns: Option<&HashSet<String>>,
) -> (String, HashMap<String, String>) {
    let mut mapping: HashMap<String, String> = HashMap::new();
    let mut result = text.to_string();
    let mut counter = 0usize;

    // Protect known proper nouns first (longest match wins)
    if let Some(nouns) = proper_nouns {
        // Sort by length descending to match longest first
        let mut sorted: Vec<&String> = nouns.iter().collect();
        sorted.sort_by_key(|a| std::cmp::Reverse(a.len()));

        for name in &sorted {
            let escaped = regex_escape(name);
            // Use word boundary matching
            let re = Regex::new(&format!(r"\b{}\b", escaped)).unwrap();
            // Replace one at a time to handle each occurrence properly
            while let Some(mat) = re.find(&result) {
                let key = format!("__PN{counter}__");
                mapping.insert(key.clone(), mat.as_str().to_string());
                // Replace only this match
                let start = mat.start();
                let end = mat.end();
                result = format!("{}{}{}", &result[..start], key, &result[end..]);
                counter += 1;
            }
        }
    }

    // Then apply regex-based placeholder patterns
    for pp in placeholder_patterns() {
        while let Some(mat) = pp.pattern.find(&result) {
            let key = format!("__{}{}__", pp.tag, counter);
            mapping.insert(key.clone(), mat.as_str().to_string());
            let start = mat.start();
            let end = mat.end();
            result = format!("{}{}{}", &result[..start], key, &result[end..]);
            counter += 1;
        }
    }

    (result, mapping)
}

/// Restore placeholder tags with their original values.
pub fn restore_placeholders(text: &str, mapping: &HashMap<String, String>) -> String {
    let mut result = text.to_string();
    for (key, original) in mapping {
        result = result.replace(key, original);
    }
    result
}

/// Apply fallback logic: replace empty or suspiciously-short translations
/// with the originals.
pub fn apply_fallbacks(
    originals: &[String],
    translations: &[String],
    skip_indices: Option<&HashSet<usize>>,
    cached_translations: Option<&HashMap<usize, String>>,
) -> Vec<String> {
    let empty_skip = HashSet::new();
    let empty_cache = HashMap::new();
    let skip = skip_indices.unwrap_or(&empty_skip);
    let cache = cached_translations.unwrap_or(&empty_cache);

    let mut result = Vec::with_capacity(originals.len());
    for (i, (original, translated)) in originals.iter().zip(translations.iter()).enumerate() {
        if skip.contains(&i) {
            result.push(cache.get(&i).cloned().unwrap_or_else(|| translated.clone()));
            continue;
        }

        let stripped = translated.trim();
        if stripped.is_empty() {
            warn!(
                "Empty translation for line {}: \"{}\" — using original as fallback",
                i, original
            );
            result.push(original.clone());
        } else if stripped.len() < 2 && original.trim().len() > 5 {
            warn!(
                "Suspiciously short translation for line {}: \"{}\" -> \"{}\" — using original as fallback",
                i, original, translated
            );
            result.push(original.clone());
        } else {
            result.push(translated.clone());
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phrase_lookup_single_word() {
        let (result, mapped) = preprocess_for_translation("Hello");
        assert!(mapped);
        assert_eq!(result, "Cześć");
    }

    #[test]
    fn test_phrase_lookup_multi_word() {
        // "Thank you" has capital T -> Title case formatting
        let (result, mapped) = preprocess_for_translation("Thank you");
        assert!(mapped);
        assert_eq!(result, "Dziękuję");
    }

    #[test]
    fn test_phrase_lookup_multi_word_lowercase() {
        // All lowercase -> no capitalization
        let (result, mapped) = preprocess_for_translation("thank you");
        assert!(mapped);
        assert_eq!(result, "dziękuję");
    }

    #[test]
    fn test_phrase_lookup_multi_word_title_case() {
        let (result, mapped) = preprocess_for_translation("Thank You");
        assert!(mapped);
        // Title case preserved
        assert_eq!(result, "Dziękuję");
    }

    #[test]
    fn test_phrase_lookup_uppercase() {
        let (result, mapped) = preprocess_for_translation("HELLO");
        assert!(mapped);
        assert_eq!(result, "CZEŚĆ");
    }

    #[test]
    fn test_not_mapped() {
        let (result, mapped) = preprocess_for_translation("This is a test sentence.");
        assert!(!mapped);
        assert_eq!(result, "This is a test sentence.");
    }

    #[test]
    fn test_placeholder_protection() {
        let text = "Call 555-1234 or visit https://example.com";
        let (result, mapping) = extract_placeholders(text, None);
        assert_ne!(result, text);
        assert!(!mapping.is_empty());

        // Restore
        let restored = restore_placeholders(&result, &mapping);
        assert_eq!(restored, text);
    }

    #[test]
    fn test_proper_noun_protection() {
        let mut nouns = HashSet::new();
        nouns.insert("Guts".to_string());
        nouns.insert("Griffith".to_string());

        let text = "Guts swung his sword at Griffith.";
        let (result, mapping) = extract_placeholders(text, Some(&nouns));
        assert!(result.contains("__PN"), "proper nouns should be replaced");
        assert!(!result.contains("Guts"), "Guts should be replaced");
        assert!(!result.contains("Griffith"), "Griffith should be replaced");

        let restored = restore_placeholders(&result, &mapping);
        assert_eq!(restored, text);
    }

    #[test]
    fn test_postprocess_trim() {
        let result = postprocess_translation("  Hello world!  ");
        assert_eq!(result, "Hello world!");
    }

    #[test]
    fn test_apply_fallbacks_empty_translation() {
        let originals = vec!["Hello".to_string()];
        let translations = vec!["".to_string()];
        let result = apply_fallbacks(&originals, &translations, None, None);
        assert_eq!(result[0], "Hello");
    }

    #[test]
    fn test_apply_fallbacks_short_translation() {
        let originals = vec!["Long sentence here".to_string()];
        let translations = vec!["a".to_string()];
        let result = apply_fallbacks(&originals, &translations, None, None);
        assert_eq!(result[0], "Long sentence here");
    }

    #[test]
    fn test_apply_fallbacks_ok() {
        let originals = vec!["Hello".to_string()];
        let translations = vec!["Cześć".to_string()];
        let result = apply_fallbacks(&originals, &translations, None, None);
        assert_eq!(result[0], "Cześć");
    }
}
