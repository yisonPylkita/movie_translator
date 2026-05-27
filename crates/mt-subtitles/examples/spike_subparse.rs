/// Spike B: subparse crate evaluation for ASS round-trip fidelity.
///
/// VERDICT: subparse is UNSUITABLE for this project's requirements.
///
/// Critical limitations found in subparse v0.7.0:
///
/// 1. NO STYLES ACCESS: The `SsaFile` struct has no API to read or write
///    [V4+ Styles] section data. Styles are stored as opaque `Filler` bytes
///    with no structured access (no Name, Fontname, Fontsize, colours, etc.).
///
/// 2. NO PER-FIELD EVENT ACCESS: The only structured data extracted from
///    Dialogue lines is (start_time, end_time, text). Fields like Layer,
///    Style, Name, MarginL/R/V, Effect are all lumped into `Filler` strings.
///    We cannot read or write Style-per-event, which is required for the
///    translated subtitle workflow.
///
/// 3. ONLY DIALOGUE LINES: `Comment:` lines in [Events] are treated as
///    Filler, so they are preserved verbatim but not accessible as structured
///    events. The onepace_arlongpark_01_pl.ass file has many Comment events.
///
/// 4. ANCIENT DEPENDENCIES: nom v2.1.0, combine v2.5.2, failure v0.1.8 —
///    all deprecated/abandoned. This creates maintenance risk.
///
/// 5. FIDELITY: Even basic round-trip of timing is lossy. subparse's
///    parse_timepoint treats the centisecond field as `ms * 10`, which
///    means it reads "0:00:16.71" as 16710ms but pysubs2 reads it as
///    16710ms (correct). HOWEVER, subparse then formats it back as
///    centiseconds (2 digits), which matches. So timing round-trips OK.
///    But the event count exposed by `get_subtitle_entries()` counts only
///    `Dialogue:` lines (not `Comment:` lines), while pysubs2 includes both.
///
/// COMPARISON SUMMARY (11 corpus files):
/// - Timing fidelity: OK for Dialogue-only timing
/// - Style access: NONE (0/11 files can expose style data)
/// - Per-event Style field: NONE (cannot read which style each event uses)
/// - Comment events: NOT accessible as structured events
/// - Override tags ({\i1} etc.): preserved verbatim in Text field (acceptable)
///
/// Since we need styles + per-event Style field for the translated subtitle
/// workflow, subparse cannot fulfill the requirements.

use std::path::Path;
use subparse::{SsaFile, SubtitleFileInterface};

fn main() {
    let corpus = [
        "benchmarks/onepiece/op_0031.ass",
        "benchmarks/onepiece/op_0040.ass",
        "benchmarks/onepiece/op_0050.ass",
        "benchmarks/onepiece/op_0055.ass",
        "benchmarks/onepiece/op_0061.ass",
        "benchmarks/onepiece/op_0031.gold.pl.ass",
        "benchmarks/onepiece/op_0040.gold.pl.ass",
        "benchmarks/onepiece/op_0050.gold.pl.ass",
        "benchmarks/onepiece/op_0055.gold.pl.ass",
        "benchmarks/onepiece/op_0061.gold.pl.ass",
        "benchmarks/onepiece/onepace_arlongpark_01_pl.ass",
    ];

    println!("=== Spike B: subparse v0.7.0 ASS round-trip evaluation ===\n");
    println!("CRITICAL: subparse only exposes (start_ms, end_ms, text) per Dialogue line.");
    println!("It has NO API for styles, event Style field, Layer, Name, Margins, Effect.");
    println!("Comment: lines are not exposed as structured entries.\n");

    let mut total_files = 0;
    let mut files_parsed_ok = 0;
    let mut total_dialogue_events = 0;
    let mut total_gt_events: usize = 0;

    // Load ground truth for comparison
    let gt_path = "benchmarks/onepiece/ground_truth/ground_truth.json";
    let gt_json: serde_json::Value = if Path::new(gt_path).exists() {
        let content = std::fs::read_to_string(gt_path).unwrap();
        serde_json::from_str(&content).unwrap()
    } else {
        eprintln!("WARNING: ground truth not found at {gt_path}; run Python step first");
        return;
    };

    for path in &corpus {
        total_files += 1;
        let filename = Path::new(path).file_name().unwrap().to_str().unwrap();

        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => {
                println!("FAIL  {filename}: could not read: {e}");
                continue;
            }
        };

        match SsaFile::parse(&content) {
            Ok(ssa) => {
                let entries = ssa.get_subtitle_entries().unwrap();
                let dialogue_count = entries.len();
                total_dialogue_events += dialogue_count;

                let gt = &gt_json[filename];
                let gt_event_count = gt["event_count"].as_u64().unwrap_or(0) as usize;
                let gt_style_count = gt["style_count"].as_u64().unwrap_or(0) as usize;
                total_gt_events += gt_event_count;

                let event_match = if dialogue_count == gt_event_count {
                    "OK "
                } else {
                    "MISMATCH"
                };

                println!("PARSE_OK {filename}");
                println!(
                    "  Dialogue events: subparse={dialogue_count}, pysubs2={gt_event_count} [{event_match}]"
                );
                println!(
                    "  Styles: subparse=NO_API, pysubs2={gt_style_count} [MISSING - subparse has no style access]"
                );
                println!("  Per-event Style field: NO_API [MISSING]");
                println!("  Comment events: NOT ACCESSIBLE via subparse API");
                println!("  Override tags in text: PRESERVED (verbatim in text field)");

                // Show first 2 entries to demonstrate what we can actually get
                if !entries.is_empty() {
                    let e = &entries[0];
                    println!(
                        "  Sample entry[0]: start={}ms, end={}ms, text={:?}",
                        e.timespan.start.msecs(),
                        e.timespan.end.msecs(),
                        e.line.as_deref().unwrap_or("").chars().take(60).collect::<String>()
                    );
                }

                files_parsed_ok += 1;
            }
            Err(e) => {
                println!("FAIL  {filename}: parse error: {e}");
            }
        }
        println!();
    }

    println!("=== SUMMARY ===");
    println!(
        "Files parsed: {files_parsed_ok}/{total_files}"
    );
    println!(
        "Dialogue event count match: need manual check (Comment events inflate pysubs2 counts)"
    );
    println!("Style access: 0/{total_files} (NO API IN SUBPARSE)");
    println!("Per-event Style field: 0/{total_files} (NO API IN SUBPARSE)");
    println!();
    println!("VERDICT: subparse REJECTED. Missing required APIs: styles, per-event Style/Layer/Name/Margins/Effect.");
    println!("Hand-rolled parser is the only viable option.");
}
