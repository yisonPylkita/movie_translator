//! PGS (Presentation Graphic Stream) subtitle stream parser.
//!
//! Parses the binary PGS format used in Blu-ray subtitles (codec:
//! `hdmv_pgs_subtitle`).  Extracts bitmap subtitle events with their
//! presentation timestamps.  The caller is responsible for OCR-ing the
//! rendered grayscale images (e.g. via Apple Vision through PyO3).
//!
//! Format reference:
//! - PGS segments have a 13-byte header (magic `PG` + PTS + DTS + type + len)
//! - Segment types: PCS (0x16), WDS (0x17), PDS (0x14), ODS (0x15), END (0x80)
//! - Bitmap data is RLE-compressed palette-indexed pixel data

use mt_core::BoundingBox;

// ── Segment types ───────────────────────────────────────────────────────────

const SEG_PCS: u8 = 0x16; // Presentation Composition Segment
const SEG_PDS: u8 = 0x14; // Palette Definition Segment
const SEG_ODS: u8 = 0x15; // Object Definition Segment

/// A single subtitle event: a rendered grayscale image at a given timestamp.
#[derive(Debug, Clone)]
pub struct PgsSubtitleEvent {
    /// Presentation timestamp in milliseconds.
    pub pts_ms: f64,
    /// Grayscale pixel data (1 byte per pixel).
    pub pixels: Vec<u8>,
    /// Width of the bitmap in pixels.
    pub width: u16,
    /// Height of the bitmap in pixels.
    pub height: u16,
}

/// Parse raw PGS `.sup` file bytes into subtitle events.
///
/// Returns a list of `(pts_ms, pixels, width, height)` tuples, one per
/// subtitle display event.  Each pixel value is grayscale (0-255).
pub fn parse_sup(data: &[u8]) -> Vec<PgsSubtitleEvent> {
    let segments = parse_segments(data);
    extract_subtitle_images(&segments)
}

// ── Low-level segment parsing ───────────────────────────────────────────────

struct Segment {
    pts: f64, // ms
    seg_type: u8,
    data: Vec<u8>, // segment payload
}

fn parse_segments(data: &[u8]) -> Vec<Segment> {
    let mut segments = Vec::new();
    let mut pos = 0;

    while pos + 13 <= data.len() {
        // Check magic
        if data[pos] != b'P' || data[pos + 1] != b'G' {
            break;
        }

        let pts_raw =
            u32::from_be_bytes([data[pos + 2], data[pos + 3], data[pos + 4], data[pos + 5]]);
        // PTS is in 90kHz ticks
        let pts = pts_raw as f64 / 90.0;

        let seg_type = data[pos + 10];
        let seg_size = u16::from_be_bytes([data[pos + 11], data[pos + 12]]) as usize;

        if seg_size > data.len().saturating_sub(pos + 13) {
            break; // malformed, stop
        }

        let seg_data = data[pos + 13..pos + 13 + seg_size].to_vec();
        segments.push(Segment {
            pts,
            seg_type,
            data: seg_data,
        });

        pos += 13 + seg_size;
    }

    segments
}

// ── RLE bitmap decoder ──────────────────────────────────────────────────────

fn decode_rle(data: &[u8], width: u16, height: u16) -> Vec<u8> {
    let total = width as usize * height as usize;
    let mut pixels = Vec::with_capacity(total);
    let mut i = 0;

    while i < data.len() && pixels.len() < total {
        let byte = data[i];
        i += 1;

        if byte != 0 {
            // Single pixel of color `byte`
            pixels.push(byte);
        } else {
            // Escape sequence starting with 0x00
            if i >= data.len() {
                break;
            }
            let flag = data[i];
            i += 1;

            if flag == 0 {
                // End of line — pad to width boundary
                while pixels.len() % width as usize != 0 && pixels.len() < total {
                    pixels.push(0);
                }
            } else if flag & 0xC0 == 0x40 {
                // Next-N-byte: next byte is length, fill with 0
                // Upper bits (0x40) indicate mode, lower 6 bits are high bits of length
                if i >= data.len() {
                    break;
                }
                let length = ((flag & 0x3F) as usize) << 8 | data[i] as usize;
                i += 1;
                for _ in 0..length {
                    if pixels.len() >= total {
                        break;
                    }
                    pixels.push(0);
                }
            } else if flag & 0xC0 == 0x80 {
                // Run of same color: lower 6 bits = length, next byte = color
                let length = (flag & 0x3F) as usize;
                if i >= data.len() {
                    break;
                }
                let color = data[i];
                i += 1;
                for _ in 0..length {
                    if pixels.len() >= total {
                        break;
                    }
                    pixels.push(color);
                }
            } else if flag & 0xC0 == 0xC0 {
                // Long run of same color: lower 6 bits + next byte = length, next+1 byte = color
                if i + 1 >= data.len() {
                    break;
                }
                let length = ((flag & 0x3F) as usize) << 8 | data[i] as usize;
                i += 1;
                let color = data[i];
                i += 1;
                for _ in 0..length {
                    if pixels.len() >= total {
                        break;
                    }
                    pixels.push(color);
                }
            } else {
                // Next-N-byte (flag & 0x3F = length), fill with 0
                let length = (flag & 0x3F) as usize;
                for _ in 0..length {
                    if pixels.len() >= total {
                        break;
                    }
                    pixels.push(0);
                }
            }
        }
    }

    // Pad to exact size
    while pixels.len() < total {
        pixels.push(0);
    }

    pixels.truncate(total);
    pixels
}

// ── Image extraction from segments ─────────────────────────────────────────

fn extract_subtitle_images(segments: &[Segment]) -> Vec<PgsSubtitleEvent> {
    // Palette tables (updated by PDS segments)
    let mut y_lut = [0u8; 256];
    let mut a_lut = [0u8; 256];

    let mut results: Vec<PgsSubtitleEvent> = Vec::new();

    let mut ods_data: Vec<u8> = Vec::new();
    let mut ods_width: u16 = 0;
    let mut ods_height: u16 = 0;
    let mut current_pts: f64 = 0.0;

    for seg in segments {
        match seg.seg_type {
            SEG_PCS => {
                current_pts = seg.pts;
                let d = &seg.data;
                let num_objects = if d.len() > 8 { d[8] } else { 0 };
                if num_objects == 0 {
                    continue;
                }
                ods_data.clear();
            }

            SEG_PDS => {
                // Palette entry: palette_id(1) + version(1) + { entry_id(1), Y(1), Cr(1), Cb(1), alpha(1) }*
                let d = &seg.data;
                let mut i = 2; // skip palette_id + version
                while i + 4 < d.len() {
                    let entry_id = d[i] as usize;
                    if entry_id < 256 {
                        y_lut[entry_id] = d[i + 1];
                        a_lut[entry_id] = d[i + 4];
                    }
                    i += 5;
                }
            }

            SEG_ODS => {
                let d = &seg.data;
                if d.len() < 4 {
                    continue;
                }
                let seq_flag = d[3];
                if seq_flag & 0x80 != 0 {
                    // First in sequence
                    if d.len() < 11 {
                        continue;
                    }
                    ods_width = u16::from_be_bytes([d[7], d[8]]);
                    ods_height = u16::from_be_bytes([d[9], d[10]]);
                    ods_data = d[11..].to_vec();
                } else {
                    // Continuation
                    if d.len() > 4 {
                        ods_data.extend_from_slice(&d[4..]);
                    }
                }

                if seq_flag & 0x40 != 0 && ods_width > 0 && ods_height > 0 && !ods_data.is_empty() {
                    // Decode RLE
                    let indexed = decode_rle(&ods_data, ods_width, ods_height);

                    // Apply palette: map indexed → grayscale using Y + alpha
                    let mut grayscale = Vec::with_capacity(indexed.len());
                    for &idx in &indexed {
                        let y = y_lut[idx as usize];
                        let alpha = a_lut[idx as usize];
                        grayscale.push(if alpha > 128 { y } else { 0 });
                    }

                    results.push(PgsSubtitleEvent {
                        pts_ms: current_pts,
                        pixels: grayscale,
                        width: ods_width,
                        height: ods_height,
                    });
                }
            }

            _ => {}
        }
    }

    results
}

/// OCR text extracted from a single PGS subtitle image via Apple Vision.
/// This is kept in Python via PyO3 — nothing in this crate depends on it.
/// The return type is defined here for cross-crate sharing.
#[derive(Debug, Clone)]
pub struct PgsOcrResult {
    /// Timestamp in milliseconds.
    pub timestamp_ms: i64,
    /// OCR text.
    pub text: String,
    /// Bounding boxes (normalized 0-1, top-left origin).
    pub boxes: Vec<BoundingBox>,
}

/// Build dialogue lines from OCR'd PGS events: deduplicate consecutive
/// identical text into timed blocks.
pub fn build_dialogue_lines_from_ocr(results: &[PgsOcrResult]) -> Vec<mt_core::DialogueLine> {
    let mut lines: Vec<mt_core::DialogueLine> = Vec::new();
    let mut prev_text = String::new();
    let mut start_ms: i64 = 0;

    for r in results {
        if r.text != prev_text {
            if !prev_text.is_empty() && prev_text.len() > 1 {
                lines.push(mt_core::DialogueLine {
                    start_ms,
                    end_ms: r.timestamp_ms,
                    text: prev_text.clone(),
                });
            }
            start_ms = r.timestamp_ms;
            prev_text = r.text.clone();
        }
    }

    // Close final line
    if !prev_text.is_empty() && prev_text.len() > 1 {
        let last_ts = results.last().map(|r| r.timestamp_ms).unwrap_or(start_ms);
        lines.push(mt_core::DialogueLine {
            start_ms,
            end_ms: last_ts + 3000,
            text: prev_text,
        });
    }

    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal valid PGS segment: just an END marker.
    /// Should produce no events.
    fn minimal_pgs() -> Vec<u8> {
        let mut buf = Vec::new();
        // END segment: PG(2) + PTS(4) + DTS(4) + type(1) + size(2)
        buf.extend_from_slice(b"PG");
        buf.extend_from_slice(&[0u8; 8]); // PTS=0, DTS=0
        buf.push(0x80u8); // END
        buf.extend_from_slice(&0u16.to_be_bytes()); // size=0
        buf
    }

    #[test]
    fn test_empty_sup() {
        let events = parse_sup(&[]);
        assert!(events.is_empty());
    }

    #[test]
    fn test_minimal_sup() {
        let data = minimal_pgs();
        let events = parse_sup(&data);
        assert!(events.is_empty());
    }

    #[test]
    fn test_no_magic_stops() {
        let data = b"\x00\x00\x00\x00";
        let events = parse_sup(data);
        assert!(events.is_empty());
    }

    #[test]
    fn test_rle_decode_two_rows() {
        // A 2x2 image: row 0=[1,2], row 1=[3,4]
        // Each row ends with 0x0000 (EOL marker)
        let data = vec![1u8, 2, 0, 0, 3, 4, 0, 0];
        let pixels = decode_rle(&data, 2, 2);
        assert_eq!(pixels.len(), 4);
        assert_eq!(pixels[0], 1);
        assert_eq!(pixels[1], 2);
        assert_eq!(pixels[2], 3);
        assert_eq!(pixels[3], 4);
    }

    #[test]
    fn test_rle_same_color_run() {
        // 0x80 mode: length=3, color=5
        let data = vec![0x00, 0x83, 5u8];
        let pixels = decode_rle(&data, 3, 1);
        assert_eq!(pixels.len(), 3);
        assert_eq!(pixels, vec![5, 5, 5]);
    }

    #[test]
    fn test_parse_unknown_segment_ignored() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"PG");
        buf.extend_from_slice(&[0u8; 8]); // PTS=0, DTS=0
        buf.push(0x99); // Unknown type
        buf.extend_from_slice(&2u16.to_be_bytes()); // size=2
        buf.extend_from_slice(&[1u8, 2]);
        let events = parse_sup(&buf);
        assert!(events.is_empty());
    }
}
