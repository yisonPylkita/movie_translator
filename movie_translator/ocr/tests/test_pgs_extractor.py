import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from movie_translator.ocr.pgs_extractor import (
    _decode_rle,
    _extract_subtitle_images,
    _format_srt_time,
    _parse_segments,
    _write_srt,
    extract_pgs_track,
)
from movie_translator.types import BoundingBox, DialogueLine

# ---------------------------------------------------------------------------
# Helpers for building PGS binary data
# ---------------------------------------------------------------------------

_SEG_PCS = 0x16
_SEG_WDS = 0x17
_SEG_PDS = 0x14
_SEG_ODS = 0x15
_SEG_END = 0x80


def _make_pgs_segment(pts_90khz: int, seg_type: int, payload: bytes) -> bytes:
    """Build a single raw PGS segment with the on-disk header format.

    Format: 'PG' (2) | PTS (4 big-endian) | DTS (4 big-endian) |
            type (1) | size (2 big-endian) | payload
    """
    return (
        b'PG'
        + struct.pack('>I', pts_90khz)
        + struct.pack('>I', 0)  # DTS = 0
        + struct.pack('B', seg_type)
        + struct.pack('>H', len(payload))
        + payload
    )


# ---------------------------------------------------------------------------
# _format_srt_time
# ---------------------------------------------------------------------------


class TestFormatSrtTime:
    def test_zero_ms(self):
        assert _format_srt_time(0) == '00:00:00,000'

    def test_exact_one_second(self):
        assert _format_srt_time(1000) == '00:00:01,000'

    def test_exact_one_minute(self):
        assert _format_srt_time(60000) == '00:01:00,000'

    def test_exact_one_hour(self):
        assert _format_srt_time(3600000) == '01:00:00,000'

    def test_mixed_components(self):
        # 1h 23m 45s 678ms
        ms = 1 * 3600000 + 23 * 60000 + 45 * 1000 + 678
        assert _format_srt_time(ms) == '01:23:45,678'

    def test_large_value(self):
        # 99h 59m 59s 999ms
        ms = 99 * 3600000 + 59 * 60000 + 59 * 1000 + 999
        assert _format_srt_time(ms) == '99:59:59,999'

    def test_milliseconds_only(self):
        assert _format_srt_time(123) == '00:00:00,123'

    def test_one_millisecond(self):
        assert _format_srt_time(1) == '00:00:00,001'

    def test_seconds_and_milliseconds(self):
        assert _format_srt_time(12345) == '00:00:12,345'


# ---------------------------------------------------------------------------
# _write_srt
# ---------------------------------------------------------------------------


class TestWriteSrt:
    def test_writes_valid_srt(self, tmp_path):
        lines = [
            DialogueLine(1000, 3000, 'Hello world'),
            DialogueLine(4000, 6500, 'Second line'),
        ]
        output = tmp_path / 'output.srt'

        _write_srt(lines, output)

        content = output.read_text()
        assert '1\n' in content
        assert '00:00:01,000 --> 00:00:03,000' in content
        assert 'Hello world' in content
        assert '2\n' in content
        assert '00:00:04,000 --> 00:00:06,500' in content
        assert 'Second line' in content

    def test_writes_single_entry(self, tmp_path):
        lines = [DialogueLine(0, 5000, 'Only line')]
        output = tmp_path / 'single.srt'

        _write_srt(lines, output)

        content = output.read_text()
        assert '1\n' in content
        assert '00:00:00,000 --> 00:00:05,000' in content
        assert 'Only line' in content

    def test_empty_lines_produces_empty_file(self, tmp_path):
        output = tmp_path / 'empty.srt'

        _write_srt([], output)

        content = output.read_text()
        assert content == ''

    def test_sequential_numbering(self, tmp_path):
        lines = [
            DialogueLine(1000, 2000, 'A'),
            DialogueLine(3000, 4000, 'B'),
            DialogueLine(5000, 6000, 'C'),
        ]
        output = tmp_path / 'numbered.srt'

        _write_srt(lines, output)

        content = output.read_text()
        assert '1\n' in content
        assert '2\n' in content
        assert '3\n' in content

    def test_multiline_subtitle_text(self, tmp_path):
        lines = [DialogueLine(1000, 3000, 'Line one\nLine two')]
        output = tmp_path / 'multiline.srt'

        _write_srt(lines, output)

        content = output.read_text()
        assert 'Line one\nLine two' in content


# ---------------------------------------------------------------------------
# _parse_segments
# ---------------------------------------------------------------------------


class TestParseSegments:
    def test_single_segment(self):
        payload = b'\x01\x02\x03'
        pts_90khz = 9000  # 9000 / 90 = 100.0 ms
        data = _make_pgs_segment(pts_90khz, _SEG_PCS, payload)

        segments = _parse_segments(data)

        assert len(segments) == 1
        assert segments[0]['type'] == _SEG_PCS
        assert segments[0]['pts'] == pytest.approx(100.0)
        assert segments[0]['data'] == payload

    def test_multiple_segments(self):
        seg1 = _make_pgs_segment(9000, _SEG_PCS, b'\x01')
        seg2 = _make_pgs_segment(18000, _SEG_PDS, b'\x02\x03')
        seg3 = _make_pgs_segment(27000, _SEG_ODS, b'\x04\x05\x06')
        data = seg1 + seg2 + seg3

        segments = _parse_segments(data)

        assert len(segments) == 3
        assert segments[0]['type'] == _SEG_PCS
        assert segments[0]['pts'] == pytest.approx(100.0)
        assert segments[1]['type'] == _SEG_PDS
        assert segments[1]['pts'] == pytest.approx(200.0)
        assert segments[2]['type'] == _SEG_ODS
        assert segments[2]['pts'] == pytest.approx(300.0)

    def test_empty_data(self):
        segments = _parse_segments(b'')
        assert segments == []

    def test_garbage_data(self):
        segments = _parse_segments(b'not a PGS stream at all')
        assert segments == []

    def test_truncated_header_ignored(self):
        # Only 10 bytes — less than the 13-byte minimum header
        segments = _parse_segments(b'PG\x00\x00\x00\x00\x00\x00\x00\x00')
        assert segments == []

    def test_pts_conversion_from_90khz(self):
        # 90000 ticks at 90kHz = exactly 1000.0 ms
        # Use a non-empty payload so the segment exceeds the 13-byte header threshold
        data = _make_pgs_segment(90000, _SEG_PCS, b'\x00')
        segments = _parse_segments(data)

        assert len(segments) == 1
        assert segments[0]['pts'] == pytest.approx(1000.0)

    def test_large_payload(self):
        payload = b'\xab' * 1000
        data = _make_pgs_segment(0, _SEG_ODS, payload)
        segments = _parse_segments(data)

        assert len(segments) == 1
        assert len(segments[0]['data']) == 1000

    def test_segment_preserves_payload_bytes(self):
        payload = bytes(range(256))
        data = _make_pgs_segment(0, _SEG_PDS, payload)
        segments = _parse_segments(data)

        assert segments[0]['data'] == payload


# ---------------------------------------------------------------------------
# _decode_rle
# ---------------------------------------------------------------------------


class TestDecodeRle:
    def test_literal_pixels(self):
        # Non-zero bytes are literal palette indices
        data = bytes([1, 2, 3, 0, 0])  # 3 pixels + end-of-line
        img = _decode_rle(data, 3, 1)

        assert img.shape == (1, 3)
        np.testing.assert_array_equal(img[0], [1, 2, 3])

    def test_end_of_line_padding(self):
        # 0x00, 0x00 = end of line, pads with zeros to width
        data = bytes([5, 0, 0])
        img = _decode_rle(data, 4, 1)

        assert img.shape == (1, 4)
        np.testing.assert_array_equal(img[0], [5, 0, 0, 0])

    def test_short_run_of_zeros(self):
        # 0x00, then flag < 0x40 → run of (flag & 0x3F) zeros
        # flag = 0x05 → 5 zeros
        data = bytes([0, 5])
        img = _decode_rle(data, 5, 1)

        assert img.shape == (1, 5)
        np.testing.assert_array_equal(img[0], [0, 0, 0, 0, 0])

    def test_long_run_of_zeros(self):
        # 0x00, flag with 0x40 bit set → length = ((flag & 0x3F) << 8) | next_byte
        # flag = 0x41, next = 0x00 → length = (1 << 8) | 0 = 256
        data = bytes([0, 0x41, 0x00])
        img = _decode_rle(data, 256, 1)

        assert img.shape == (1, 256)
        assert np.all(img == 0)

    def test_short_run_of_color(self):
        # 0x00, flag with 0x80 bit set → length = flag & 0x3F, color = next byte
        # flag = 0x83 → length 3, color = 0xFF
        data = bytes([0, 0x83, 0xFF])
        img = _decode_rle(data, 3, 1)

        assert img.shape == (1, 3)
        np.testing.assert_array_equal(img[0], [255, 255, 255])

    def test_long_run_of_color(self):
        # 0x00, flag with 0xC0 bits set → length = ((flag & 0x3F) << 8) | next, color = next
        # flag = 0xC1, next_len = 0x00 → length = 256, color = 0x0A
        data = bytes([0, 0xC1, 0x00, 0x0A])
        img = _decode_rle(data, 256, 1)

        assert img.shape == (1, 256)
        assert np.all(img == 10)

    def test_multi_row(self):
        # Two rows: [1, 2] then [3, 4]
        # Each row ends with 0x00 0x00 (end-of-line)
        data = bytes([1, 2, 0, 0, 3, 4, 0, 0])
        img = _decode_rle(data, 2, 2)

        assert img.shape == (2, 2)
        np.testing.assert_array_equal(img[0], [1, 2])
        np.testing.assert_array_equal(img[1], [3, 4])

    def test_short_data_pads_with_zeros(self):
        # If RLE data runs out before filling the bitmap, pad with 0
        data = bytes([7])
        img = _decode_rle(data, 4, 2)

        assert img.shape == (2, 4)
        assert img[0, 0] == 7
        # Rest should be zero-padded
        assert np.sum(img) == 7

    def test_empty_data(self):
        img = _decode_rle(b'', 2, 2)

        assert img.shape == (2, 2)
        assert np.all(img == 0)


# ---------------------------------------------------------------------------
# _extract_subtitle_images
# ---------------------------------------------------------------------------


class TestExtractSubtitleImages:
    @staticmethod
    def _make_pcs_data(num_objects: int = 1) -> bytes:
        """Build minimal PCS payload with the given object count at byte 8."""
        # PCS data: first 8 bytes can be anything, byte[8] = num_objects
        return b'\x00' * 8 + bytes([num_objects])

    @staticmethod
    def _make_pds_data(entries: list[tuple[int, int, int]]) -> bytes:
        """Build PDS payload. entries = [(id, Y, alpha), ...]."""
        # 2-byte header (palette_id + version), then 5 bytes per entry
        # Format per entry: id, Y, Cr, Cb, A
        result = b'\x00\x00'
        for entry_id, y_val, alpha in entries:
            result += bytes([entry_id, y_val, 128, 128, alpha])
        return result

    @staticmethod
    def _make_ods_data(width: int, height: int, rle_data: bytes) -> bytes:
        """Build ODS payload for a single (first+last) object."""
        # ODS header: object_id (2), version (1), seq_flag (1), data_len (3),
        # width (2), height (2), then RLE data
        # seq_flag = 0xC0 means first-and-last
        data_len = len(rle_data) + 4  # includes width/height/rle
        return (
            b'\x00\x00'  # object_id
            + b'\x00'  # version
            + bytes([0xC0])  # seq_flag: first + last
            + struct.pack('>I', data_len)[1:]  # 3-byte big-endian data_len
            + struct.pack('>H', width)
            + struct.pack('>H', height)
            + rle_data
        )

    def test_extracts_single_image(self):
        # A 2x1 image with palette index 1 → Y=200, alpha=255
        pcs = {'pts': 100.0, 'type': _SEG_PCS, 'data': self._make_pcs_data(1)}
        pds = {
            'pts': 100.0,
            'type': _SEG_PDS,
            'data': self._make_pds_data([(1, 200, 255)]),
        }
        # RLE: two literal pixels (index 1), end-of-line
        rle_data = bytes([1, 1, 0, 0])
        ods = {'pts': 100.0, 'type': _SEG_ODS, 'data': self._make_ods_data(2, 1, rle_data)}

        results = _extract_subtitle_images([pcs, pds, ods])

        assert len(results) == 1
        pts, img, w, h = results[0]
        assert pts == pytest.approx(100.0)
        assert w == 2
        assert h == 1
        assert img.shape == (1, 2)
        # Both pixels should be Y=200 (palette index 1, alpha=255 > 128)
        np.testing.assert_array_equal(img[0], [200, 200])

    def test_skips_clear_event(self):
        # PCS with num_objects=0 is a clear event
        pcs_clear = {'pts': 50.0, 'type': _SEG_PCS, 'data': self._make_pcs_data(0)}

        results = _extract_subtitle_images([pcs_clear])

        assert len(results) == 0

    def test_transparent_pixels_become_black(self):
        # Palette entry with alpha < 128 → pixel should become 0
        pcs = {'pts': 100.0, 'type': _SEG_PCS, 'data': self._make_pcs_data(1)}
        pds = {
            'pts': 100.0,
            'type': _SEG_PDS,
            'data': self._make_pds_data([(1, 200, 50)]),  # alpha=50 < 128
        }
        rle_data = bytes([1, 0, 0])  # one pixel index 1, pad to width 2
        ods = {'pts': 100.0, 'type': _SEG_ODS, 'data': self._make_ods_data(2, 1, rle_data)}

        results = _extract_subtitle_images([pcs, pds, ods])

        assert len(results) == 1
        _, img, _, _ = results[0]
        # Pixel with alpha < 128 should be 0
        assert img[0, 0] == 0

    def test_multiple_display_sets(self):
        pcs1 = {'pts': 100.0, 'type': _SEG_PCS, 'data': self._make_pcs_data(1)}
        pds1 = {
            'pts': 100.0,
            'type': _SEG_PDS,
            'data': self._make_pds_data([(1, 200, 255)]),
        }
        rle1 = bytes([1, 0, 0])
        ods1 = {'pts': 100.0, 'type': _SEG_ODS, 'data': self._make_ods_data(1, 1, rle1)}

        pcs2 = {'pts': 500.0, 'type': _SEG_PCS, 'data': self._make_pcs_data(1)}
        # PDS not repeated — palette carries over
        rle2 = bytes([1, 0, 0])
        ods2 = {'pts': 500.0, 'type': _SEG_ODS, 'data': self._make_ods_data(1, 1, rle2)}

        results = _extract_subtitle_images([pcs1, pds1, ods1, pcs2, ods2])

        assert len(results) == 2
        assert results[0][0] == pytest.approx(100.0)
        assert results[1][0] == pytest.approx(500.0)

    def test_empty_segments(self):
        results = _extract_subtitle_images([])
        assert results == []

    def test_palette_update_applies_to_subsequent_images(self):
        # First display set with palette entry 1 → Y=100
        pcs1 = {'pts': 100.0, 'type': _SEG_PCS, 'data': self._make_pcs_data(1)}
        pds1 = {
            'pts': 100.0,
            'type': _SEG_PDS,
            'data': self._make_pds_data([(1, 100, 255)]),
        }
        rle1 = bytes([1, 0, 0])
        ods1 = {'pts': 100.0, 'type': _SEG_ODS, 'data': self._make_ods_data(1, 1, rle1)}

        # Second display set updates palette entry 1 → Y=250
        pcs2 = {'pts': 500.0, 'type': _SEG_PCS, 'data': self._make_pcs_data(1)}
        pds2 = {
            'pts': 500.0,
            'type': _SEG_PDS,
            'data': self._make_pds_data([(1, 250, 255)]),
        }
        rle2 = bytes([1, 0, 0])
        ods2 = {'pts': 500.0, 'type': _SEG_ODS, 'data': self._make_ods_data(1, 1, rle2)}

        results = _extract_subtitle_images([pcs1, pds1, ods1, pcs2, pds2, ods2])

        assert len(results) == 2
        # First image uses Y=100
        assert results[0][1][0, 0] == 100
        # Second image uses updated Y=250
        assert results[1][1][0, 0] == 250


# ---------------------------------------------------------------------------
# _ocr_grayscale_image (mocked — platform-dependent)
# ---------------------------------------------------------------------------


class TestOcrGrayscaleImage:
    @patch('movie_translator.ocr.pgs_extractor._VISION_AVAILABLE', False)
    def test_returns_empty_when_vision_unavailable(self):
        from movie_translator.ocr.pgs_extractor import _ocr_grayscale_image

        img = np.zeros((10, 10), dtype=np.uint8)
        text, boxes = _ocr_grayscale_image(img)

        assert text == ''
        assert boxes == []

    @patch('movie_translator.ocr.pgs_extractor._VISION_AVAILABLE', True)
    @patch('movie_translator.ocr.pgs_extractor.Quartz')
    @patch('movie_translator.ocr.pgs_extractor.Vision')
    def test_calls_vision_framework(self, mock_vision, mock_quartz):
        from movie_translator.ocr.pgs_extractor import _ocr_grayscale_image

        # Set up mock chain for Vision
        mock_cg_image = MagicMock()
        mock_quartz.CGColorSpaceCreateDeviceGray.return_value = MagicMock()
        mock_quartz.CGDataProviderCreateWithData.return_value = MagicMock()
        mock_quartz.CGImageCreate.return_value = mock_cg_image

        mock_request = MagicMock()
        mock_vision.VNRecognizeTextRequest.alloc.return_value.init.return_value = mock_request
        mock_vision.VNRequestTextRecognitionLevelAccurate = 1

        mock_handler = MagicMock()
        mock_vision.VNImageRequestHandler.alloc.return_value.initWithCGImage_options_.return_value = mock_handler
        mock_handler.performRequests_error_.return_value = (True, None)

        # Mock OCR results: one observation with text "Hello"
        mock_obs = MagicMock()
        mock_candidate = MagicMock()
        mock_candidate.string.return_value = 'Hello'
        mock_obs.topCandidates_.return_value = [mock_candidate]
        mock_bbox = MagicMock()
        mock_bbox.origin.x = 0.1
        mock_bbox.origin.y = 0.2
        mock_bbox.size.width = 0.8
        mock_bbox.size.height = 0.1
        mock_obs.boundingBox.return_value = mock_bbox
        mock_request.results.return_value = [mock_obs]

        img = np.ones((20, 100), dtype=np.uint8) * 200
        text, boxes = _ocr_grayscale_image(img)

        assert text == 'Hello'
        assert len(boxes) == 1
        assert boxes[0].x == pytest.approx(0.1)
        assert boxes[0].y == pytest.approx(0.7)
        assert boxes[0].width == pytest.approx(0.8)
        assert boxes[0].height == pytest.approx(0.1)

    @patch('movie_translator.ocr.pgs_extractor._VISION_AVAILABLE', True)
    @patch('movie_translator.ocr.pgs_extractor.Quartz')
    @patch('movie_translator.ocr.pgs_extractor.Vision')
    def test_returns_empty_on_failed_cg_image(self, mock_vision, mock_quartz):
        from movie_translator.ocr.pgs_extractor import _ocr_grayscale_image

        mock_quartz.CGColorSpaceCreateDeviceGray.return_value = MagicMock()
        mock_quartz.CGDataProviderCreateWithData.return_value = MagicMock()
        mock_quartz.CGImageCreate.return_value = None  # CGImage creation failed

        img = np.zeros((10, 10), dtype=np.uint8)
        text, boxes = _ocr_grayscale_image(img)

        assert text == ''
        assert boxes == []


# ---------------------------------------------------------------------------
# extract_pgs_track (public API — mocked subprocess + OCR)
# ---------------------------------------------------------------------------


class TestExtractPgsTrack:
    @patch('movie_translator.ocr.pgs_extractor.is_ocr_available', return_value=False)
    def test_returns_none_when_ocr_unavailable(self, mock_ocr):
        result = extract_pgs_track(Path('/fake/video.mkv'), 2, Path('/tmp/work'))
        assert result is None

    @patch('movie_translator.ocr.pgs_extractor.is_ocr_available', return_value=True)
    @patch('movie_translator.ocr.pgs_extractor.subprocess.run')
    def test_returns_none_on_mkvextract_failure(self, mock_run, mock_ocr, tmp_path):
        mock_run.return_value = MagicMock(returncode=1, stderr='error')

        result = extract_pgs_track(Path('/fake/video.mkv'), 2, tmp_path)

        assert result is None

    @patch('movie_translator.ocr.pgs_extractor._ocr_grayscale_image')
    @patch('movie_translator.ocr.pgs_extractor.is_ocr_available', return_value=True)
    @patch('movie_translator.ocr.pgs_extractor.subprocess.run')
    def test_full_pipeline(self, mock_run, mock_ocr, mock_ocr_image, tmp_path):
        video_path = tmp_path / 'video.mkv'
        video_path.touch()

        # Build a minimal PGS .sup file with one subtitle image
        pcs_data = b'\x00' * 8 + bytes([1])  # num_objects=1
        pds_entries = bytes(
            [
                0x00,
                0x00,  # palette header
                1,
                200,
                128,
                128,
                255,  # entry 1: Y=200, alpha=255
            ]
        )
        ods_rle = bytes([1, 1, 1, 1, 0, 0])  # 4 pixels of color 1
        ods_data = (
            b'\x00\x00\x00'
            + bytes([0xC0])
            + struct.pack('>I', len(ods_rle) + 4)[1:]
            + struct.pack('>H', 4)
            + struct.pack('>H', 1)
            + ods_rle
        )

        pgs_data = (
            _make_pgs_segment(9000, _SEG_PCS, pcs_data)
            + _make_pgs_segment(9000, _SEG_PDS, pds_entries)
            + _make_pgs_segment(9000, _SEG_ODS, ods_data)
            + _make_pgs_segment(9000, _SEG_END, b'')
        )

        def fake_mkvextract(cmd, **kwargs):
            # Write the .sup file that mkvextract would produce
            sup_path = tmp_path / 'pgs_ocr' / 'track.sup'
            sup_path.parent.mkdir(parents=True, exist_ok=True)
            sup_path.write_bytes(pgs_data)
            return MagicMock(returncode=0, stderr='')

        mock_run.side_effect = fake_mkvextract
        mock_ocr_image.return_value = ('Hello world', [BoundingBox(0.1, 0.1, 0.8, 0.1)])

        result = extract_pgs_track(video_path, 2, tmp_path)

        assert result is not None
        assert result.exists()
        assert result.suffix == '.srt'
        content = result.read_text()
        assert 'Hello world' in content
        assert '00:00:00' in content

    @patch('movie_translator.ocr.pgs_extractor._ocr_grayscale_image')
    @patch('movie_translator.ocr.pgs_extractor.is_ocr_available', return_value=True)
    @patch('movie_translator.ocr.pgs_extractor.subprocess.run')
    def test_returns_none_when_ocr_produces_no_text(
        self, mock_run, mock_ocr, mock_ocr_image, tmp_path
    ):
        video_path = tmp_path / 'video.mkv'
        video_path.touch()

        # Minimal PGS with one subtitle image
        pcs_data = b'\x00' * 8 + bytes([1])
        pds_data = b'\x00\x00' + bytes([1, 200, 128, 128, 255])
        rle = bytes([1, 0, 0])
        ods_data = (
            b'\x00\x00\x00'
            + bytes([0xC0])
            + struct.pack('>I', len(rle) + 4)[1:]
            + struct.pack('>H', 1)
            + struct.pack('>H', 1)
            + rle
        )

        pgs_data = (
            _make_pgs_segment(9000, _SEG_PCS, pcs_data)
            + _make_pgs_segment(9000, _SEG_PDS, pds_data)
            + _make_pgs_segment(9000, _SEG_ODS, ods_data)
        )

        def fake_mkvextract(cmd, **kwargs):
            sup_path = tmp_path / 'pgs_ocr' / 'track.sup'
            sup_path.parent.mkdir(parents=True, exist_ok=True)
            sup_path.write_bytes(pgs_data)
            return MagicMock(returncode=0, stderr='')

        mock_run.side_effect = fake_mkvextract
        mock_ocr_image.return_value = ('', [])  # OCR produces no text

        result = extract_pgs_track(video_path, 2, tmp_path)

        assert result is None

    @patch('movie_translator.ocr.pgs_extractor.is_ocr_available', return_value=True)
    @patch('movie_translator.ocr.pgs_extractor.subprocess.run')
    def test_returns_none_when_no_images_found(self, mock_run, mock_ocr, tmp_path):
        video_path = tmp_path / 'video.mkv'
        video_path.touch()

        # PGS data with only clear events (num_objects=0)
        pcs_data = b'\x00' * 8 + bytes([0])
        pgs_data = _make_pgs_segment(9000, _SEG_PCS, pcs_data)

        def fake_mkvextract(cmd, **kwargs):
            sup_path = tmp_path / 'pgs_ocr' / 'track.sup'
            sup_path.parent.mkdir(parents=True, exist_ok=True)
            sup_path.write_bytes(pgs_data)
            return MagicMock(returncode=0, stderr='')

        mock_run.side_effect = fake_mkvextract

        result = extract_pgs_track(video_path, 2, tmp_path)

        assert result is None

    @patch('movie_translator.ocr.pgs_extractor._ocr_grayscale_image')
    @patch('movie_translator.ocr.pgs_extractor.is_ocr_available', return_value=True)
    @patch('movie_translator.ocr.pgs_extractor.subprocess.run')
    def test_deduplicates_consecutive_identical_text(
        self, mock_run, mock_ocr, mock_ocr_image, tmp_path
    ):
        video_path = tmp_path / 'video.mkv'
        video_path.touch()

        # Build two subtitle images at different PTS values
        pcs_data = b'\x00' * 8 + bytes([1])
        pds_data = b'\x00\x00' + bytes([1, 200, 128, 128, 255])
        rle = bytes([1, 0, 0])
        ods_data = (
            b'\x00\x00\x00'
            + bytes([0xC0])
            + struct.pack('>I', len(rle) + 4)[1:]
            + struct.pack('>H', 1)
            + struct.pack('>H', 1)
            + rle
        )

        pgs_data = b''
        for pts in (9000, 18000, 27000):
            pgs_data += _make_pgs_segment(pts, _SEG_PCS, pcs_data)
            pgs_data += _make_pgs_segment(pts, _SEG_PDS, pds_data)
            pgs_data += _make_pgs_segment(pts, _SEG_ODS, ods_data)

        # Third event is a clear
        pgs_data += _make_pgs_segment(36000, _SEG_PCS, b'\x00' * 8 + bytes([0]))

        def fake_mkvextract(cmd, **kwargs):
            sup_path = tmp_path / 'pgs_ocr' / 'track.sup'
            sup_path.parent.mkdir(parents=True, exist_ok=True)
            sup_path.write_bytes(pgs_data)
            return MagicMock(returncode=0, stderr='')

        mock_run.side_effect = fake_mkvextract

        call_count = [0]

        def ocr_side_effect(img):
            call_count[0] += 1
            if call_count[0] <= 2:
                return ('Same text', [BoundingBox(0.1, 0.1, 0.8, 0.1)])
            return ('Different text', [BoundingBox(0.1, 0.1, 0.8, 0.1)])

        mock_ocr_image.side_effect = ocr_side_effect

        result = extract_pgs_track(video_path, 2, tmp_path)

        assert result is not None
        content = result.read_text()
        # Should have two dialogue entries: "Same text" and "Different text"
        assert 'Same text' in content
        assert 'Different text' in content

    @patch('movie_translator.ocr.pgs_extractor._ocr_grayscale_image')
    @patch('movie_translator.ocr.pgs_extractor.is_ocr_available', return_value=True)
    @patch('movie_translator.ocr.pgs_extractor.subprocess.run')
    def test_cleans_up_sup_file(self, mock_run, mock_ocr, mock_ocr_image, tmp_path):
        video_path = tmp_path / 'video.mkv'
        video_path.touch()

        pcs_data = b'\x00' * 8 + bytes([1])
        pds_data = b'\x00\x00' + bytes([1, 200, 128, 128, 255])
        rle = bytes([1, 0, 0])
        ods_data = (
            b'\x00\x00\x00'
            + bytes([0xC0])
            + struct.pack('>I', len(rle) + 4)[1:]
            + struct.pack('>H', 1)
            + struct.pack('>H', 1)
            + rle
        )
        pgs_data = (
            _make_pgs_segment(9000, _SEG_PCS, pcs_data)
            + _make_pgs_segment(9000, _SEG_PDS, pds_data)
            + _make_pgs_segment(9000, _SEG_ODS, ods_data)
        )

        def fake_mkvextract(cmd, **kwargs):
            sup_path = tmp_path / 'pgs_ocr' / 'track.sup'
            sup_path.parent.mkdir(parents=True, exist_ok=True)
            sup_path.write_bytes(pgs_data)
            return MagicMock(returncode=0, stderr='')

        mock_run.side_effect = fake_mkvextract
        mock_ocr_image.return_value = ('Text', [BoundingBox(0.1, 0.1, 0.8, 0.1)])

        extract_pgs_track(video_path, 2, tmp_path)

        sup_path = tmp_path / 'pgs_ocr' / 'track.sup'
        assert not sup_path.exists(), '.sup file should be cleaned up after extraction'
