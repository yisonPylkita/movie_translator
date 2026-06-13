//! Generate binary masks from bounding boxes for subtitle inpainting.
//!
//! Inpainting backends mask the region to be filled.  This module converts
//! `BoundingBox` lists (normalized coordinates) into a pixel-level mask
//! where white pixels indicate regions to inpaint.

use mt_core::BoundingBox;

/// Generate a binary mask buffer (1 channel, u8) from bounding boxes.
///
/// Returns a flat `Vec<u8>` of length `width * height`, where `255` = inpaint
/// region and `0` = keep region.  The caller is responsible for wrapping this
/// into an image format appropriate for the inpainting backend.
///
/// `dilation_px` expands each box by the given number of pixels in all
/// directions to avoid hard edges at the text boundary.
pub fn generate_mask(
    boxes: &[BoundingBox],
    frame_width: u32,
    frame_height: u32,
    dilation_px: u32,
) -> Vec<u8> {
    let w = frame_width as i64;
    let h = frame_height as i64;
    let d = dilation_px as i64;

    let mut mask = vec![0u8; (w * h) as usize];

    for bbox in boxes {
        let x1 = ((bbox.x * w as f64) as i64 - d).max(0);
        let y1 = ((bbox.y * h as f64) as i64 - d).max(0);
        let x2 = (((bbox.x + bbox.width) * w as f64) as i64 + d).min(w - 1);
        let y2 = (((bbox.y + bbox.height) * h as f64) as i64 + d).min(h - 1);

        for y in y1..=y2 {
            let row_start = y * w;
            for x in x1..=x2 {
                mask[(row_start + x) as usize] = 255;
            }
        }
    }

    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_mask_single_box() {
        let boxes = vec![BoundingBox {
            x: 0.25,
            y: 0.75,
            width: 0.5,
            height: 0.1,
        }];
        let mask = generate_mask(&boxes, 100, 100, 0);
        // The box should cover roughly x=25..75, y=75..85
        let w = 100usize;
        // Check a pixel inside the box
        assert_eq!(mask[75 * w + 50], 255);
        // Check a pixel outside
        assert_eq!(mask[10 * w + 10], 0);
    }

    #[test]
    fn test_generate_mask_empty_boxes() {
        let mask = generate_mask(&[], 100, 100, 0);
        assert!(mask.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_generate_mask_dilation() {
        let boxes = vec![BoundingBox {
            x: 0.5,
            y: 0.5,
            width: 0.0,
            height: 0.0,
        }];
        // With dilation=5, a single pixel at center expands to 11x11 block
        let mask = generate_mask(&boxes, 100, 100, 5);
        let nonzero: usize = mask.iter().filter(|&&v| v > 0).count();
        // Should be roughly 11x11 = 121 pixels (minus clipping)
        assert!(nonzero >= 100, "expected ~121 nonzero, got {nonzero}");
    }
}
