//! Apple Translation backend — on-device EN->PL via macOS Translation framework.
//!
//! Calls the compiled Swift `translate_bridge` binary via subprocess, passing
//! JSON on stdin and receiving translated texts on stdout.  Sentence merging,
//! placeholder protection, and postprocessing are handled in Rust (see
//! `mt-subtitles::sentence_merger` and `mt-subtitles::enhancements`).

use std::collections::{HashMap, HashSet};
use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use mt_core::MtError;
use mt_core::swift_bridge::{ensure_compiled, macos_at_least};
use mt_core::{DialogueLine, Result as MtResult};
use mt_subtitles::enhancements::{
    PLACEHOLDER_ONLY_RE, apply_fallbacks, extract_placeholders, postprocess_translation,
    preprocess_for_translation, restore_placeholders,
};
use mt_subtitles::sentence_merger::{merge_for_translation, unmerge_translations};
use serde::{Deserialize, Serialize};
use serde_json::{from_slice, to_string};
use tracing::info;

// ── Paths ───────────────────────────────────────────────────────────────────

/// Swift source living next to the translation package (looked up at runtime).
fn swift_source() -> PathBuf {
    // At runtime the binary is distributed; we resolve the source relative to
    // the repo root, discovered by the caller or set via MT_REPO_ROOT.
    // Try to find the source relative to the current directory / exe.
    let candidates = [
        "crates/mt-ml/swift/translate_bridge.swift",
        "../crates/mt-ml/swift/translate_bridge.swift",
        "movie_translator/translation/swift/translate_bridge.swift",
        "../movie_translator/translation/swift/translate_bridge.swift",
    ];
    for c in &candidates {
        let p = PathBuf::from(c);
        if p.exists() {
            return p;
        }
    }
    // Fallback: check MT_REPO_ROOT
    if let Ok(root) = env::var("MT_REPO_ROOT") {
        let p = PathBuf::from(root).join("crates/mt-ml/swift/translate_bridge.swift");
        if p.exists() {
            return p;
        }
    }
    // Look in the standard repo structure from the repo root
    if let Ok(cwd) = env::current_dir() {
        for ancestor in cwd.ancestors() {
            let p = ancestor.join("crates/mt-ml/swift/translate_bridge.swift");
            if p.exists() {
                return p;
            }
        }
    }
    PathBuf::from("crates/mt-ml/swift/translate_bridge.swift")
}

fn swift_binary() -> PathBuf {
    let mut bin = swift_source();
    bin.set_file_name("translate_bridge");
    // Compile outputs next to the source, same as the Python bridge
    bin.with_extension("")
}

// ── Apple Translation request/response ──────────────────────────────────────

#[derive(Serialize)]
struct TranslateRequest {
    texts: Vec<String>,
    source: String,
    target: String,
}

#[derive(Deserialize)]
struct TranslateResponse {
    #[serde(default)]
    translations: Vec<String>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    code: Option<String>,
}

// ── Cache for compiled binary ───────────────────────────────────────────────

fn ensure_translation_bridge() -> MtResult<PathBuf> {
    let source = swift_source();
    let binary = swift_binary();
    ensure_compiled(
        &source,
        &binary,
        &["-parse-as-library", "-framework", "Translation"],
        Duration::from_secs(60),
    )
}

// ── Check availability ──────────────────────────────────────────────────────

/// Check if Apple Translation is available on this system.
pub fn is_available() -> bool {
    if !macos_at_least(26) {
        return false;
    }
    if !swift_source().exists() {
        return false;
    }
    // Try compilation
    ensure_translation_bridge().is_ok()
}

// ── Core translation call ───────────────────────────────────────────────────

/// Call the Swift bridge binary with a batch of texts.
///
/// Returns translated texts (same length as input).
fn call_swift_binary(binary: &Path, texts: &[String]) -> MtResult<Vec<String>> {
    let request = TranslateRequest {
        texts: texts.to_vec(),
        source: "en".to_string(),
        target: "pl".to_string(),
    };

    let request_json = to_string(&request)
        .map_err(|e| MtError::Parse(format!("failed to serialize request: {e}")))?;

    let mut child = Command::new(binary)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(MtError::Io)?;

    // Write request JSON to stdin, then close stdin
    if let Some(mut stdin) = child.stdin.take() {
        use std::io::Write;
        stdin
            .write_all(request_json.as_bytes())
            .map_err(MtError::Io)?;
    }

    let result = child.wait_with_output().map_err(MtError::Io)?;

    if result.stdout.is_empty() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        return Err(MtError::Parse(format!(
            "Swift bridge returned no output (exit code {}): {}",
            result.status,
            truncate(&stderr, 500),
        )));
    }

    let response: TranslateResponse = from_slice(&result.stdout).map_err(|e| {
        let out = String::from_utf8_lossy(&result.stdout);
        MtError::Parse(format!(
            "Invalid JSON from Swift bridge: {e}\nOutput: {}",
            truncate(&out, 200),
        ))
    })?;

    if let Some(err) = response.error {
        return Err(MtError::Parse(format!(
            "Apple Translation error ({}): {}",
            response.code.unwrap_or_default(),
            err,
        )));
    }

    if response.translations.len() != texts.len() {
        return Err(MtError::Parse(format!(
            "Translation count mismatch: sent {}, got {}",
            texts.len(),
            response.translations.len(),
        )));
    }

    Ok(response.translations)
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("...{}", &s[s.len() - max..])
    }
}

// ── Public API ──────────────────────────────────────────────────────────────

/// Translate dialogue lines using the Apple Translation framework.
///
/// This is the Rust-native Apple backend: it handles sentence merging,
/// placeholder protection, Swift bridge calls, and postprocessing entirely
/// in Rust, eliminating the Python `apple_backend.py` wrapper.
pub fn translate(
    lines: &[DialogueLine],
    batch_size: u32,
    proper_nouns: Option<&[String]>,
) -> MtResult<Vec<DialogueLine>> {
    let binary = ensure_translation_bridge()?;
    info!("Apple Translation backend using: {}", binary.display());

    let texts: Vec<_> = lines.iter().map(|l| l.text.clone()).collect();
    let proper_set = proper_nouns.map(|n| n.iter().cloned().collect());

    // Step 1: Sentence merging
    let (merged_texts, groups) = merge_for_translation(&texts);

    // Step 2: Apply enhancements (placeholders, preprocessing)
    let mut placeholder_mappings = Vec::new();
    let mut skip_indices = HashSet::new();
    let mut cached_translations = HashMap::new();
    let mut processed_texts: Vec<String> = Vec::new();

    for (i, text) in merged_texts.iter().enumerate() {
        let (protected, mapping) = extract_placeholders(text, proper_set.as_ref());
        placeholder_mappings.push(mapping);

        // If the line is nothing but a placeholder tag + punctuation, skip model
        if PLACEHOLDER_ONLY_RE.is_match(protected.trim()) {
            let restored = restore_placeholders(&protected, &placeholder_mappings[i]);
            processed_texts.push(restored.clone());
            skip_indices.insert(i);
            cached_translations.insert(i, restored);
            continue;
        }

        let (preprocessed, was_mapped) = preprocess_for_translation(&protected);
        if was_mapped {
            let cache_text = preprocessed.clone();
            processed_texts.push(preprocessed);
            skip_indices.insert(i);
            cached_translations.insert(i, cache_text);
        } else {
            processed_texts.push(preprocessed);
        }
    }

    // Step 3: Collect texts that need actual translation
    let texts_to_translate: Vec<_> = (0..processed_texts.len())
        .filter(|i| !skip_indices.contains(i))
        .collect();
    let translate_only: Vec<_> = texts_to_translate
        .iter()
        .map(|&i| processed_texts[i].clone())
        .collect();

    let mut translations = vec![String::new(); processed_texts.len()];

    // Fill cached translations
    for (&i, cached) in &cached_translations {
        translations[i] = cached.clone();
    }

    // Step 4: Translate in batches
    if !translate_only.is_empty() {
        let bs = batch_size as usize;
        for chunk_start in (0..translate_only.len()).step_by(bs) {
            let chunk_end = (chunk_start + bs).min(translate_only.len());
            let batch: Vec<_> = translate_only[chunk_start..chunk_end].to_vec();
            let batch_results = call_swift_binary(&binary, &batch)?;
            for (j, translated) in batch_results.iter().enumerate() {
                let original_idx = texts_to_translate[chunk_start + j];
                translations[original_idx] = translated.clone();
            }
        }
    }

    // Step 5: Post-processing
    for t in &mut translations {
        *t = postprocess_translation(t);
    }

    // Step 6: Restore placeholders
    for (i, mapping) in placeholder_mappings.iter().enumerate() {
        translations[i] = restore_placeholders(&translations[i], mapping);
    }

    // Step 7: Apply fallbacks
    let final_texts = apply_fallbacks(
        &processed_texts,
        &translations,
        Some(&skip_indices),
        Some(&cached_translations),
    );

    // Step 8: Unmerge translations back to original line count
    let unmerged = unmerge_translations(&final_texts, &groups, &texts);

    // Step 9: Reconstruct DialogueLines
    Ok(lines
        .iter()
        .zip(unmerged.iter())
        .map(|(original, text)| DialogueLine {
            start_ms: original.start_ms,
            end_ms: original.end_ms,
            text: text.clone(),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_short() {
        assert_eq!(truncate("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_long() {
        let long = "a".repeat(1000);
        let result = truncate(&long, 10);
        // Returns "..." + last 10 chars = 13 chars
        assert_eq!(result.len(), 13);
        assert!(result.ends_with("aaaaaaaaaa"));
    }
}
