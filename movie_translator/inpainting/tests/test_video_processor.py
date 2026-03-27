import subprocess

import pytest

from movie_translator.ffmpeg import get_ffmpeg, probe_video_encoding
from movie_translator.inpainting.video_processor import (
    _build_subtitle_lookup,
    _compute_crop_region,
    _remap_boxes_to_crop,
    remove_burned_in_subtitles,
)
from movie_translator.types import BoundingBox, OCRResult


class TestBuildSubtitleLookup:
    def test_maps_ocr_result_to_frame_range(self):
        box = BoundingBox(x=0.1, y=0.8, width=0.8, height=0.1)
        results = [OCRResult(timestamp_ms=1000, text='Hello', boxes=[box])]

        lookup = _build_subtitle_lookup(results, fps=24.0)

        # 1000ms at 24fps = frame 24. Should cover frames 24-47 (one second)
        assert 24 in lookup
        assert 47 in lookup
        assert 23 not in lookup
        assert 48 not in lookup
        assert lookup[24] == [box]

    def test_skips_empty_text(self):
        box = BoundingBox(x=0.1, y=0.8, width=0.8, height=0.1)
        results = [
            OCRResult(timestamp_ms=0, text='', boxes=[]),
            OCRResult(timestamp_ms=1000, text='Hello', boxes=[box]),
        ]

        lookup = _build_subtitle_lookup(results, fps=24.0)

        assert 0 not in lookup
        assert 23 not in lookup
        assert 24 in lookup

    def test_consecutive_results_cover_continuous_range(self):
        box = BoundingBox(x=0.1, y=0.8, width=0.8, height=0.1)
        results = [
            OCRResult(timestamp_ms=0, text='Hello', boxes=[box]),
            OCRResult(timestamp_ms=1000, text='Hello', boxes=[box]),
        ]

        lookup = _build_subtitle_lookup(results, fps=24.0)

        for f in range(48):
            assert f in lookup

    def test_empty_results(self):
        lookup = _build_subtitle_lookup([], fps=24.0)
        assert lookup == {}

    def test_skips_results_without_boxes(self):
        results = [OCRResult(timestamp_ms=0, text='Hello', boxes=[])]

        lookup = _build_subtitle_lookup(results, fps=24.0)

        assert lookup == {}


class TestComputeCropRegion:
    def test_crops_around_subtitle_region(self):
        boxes = [BoundingBox(x=0.1, y=0.8, width=0.8, height=0.1)]
        x1, y1, x2, y2 = _compute_crop_region(boxes, 1920, 1080, padding_px=40)

        assert x1 == 0
        assert x2 == 1920
        # y1 should be box top (0.8*1080=864) minus padding (40) = 824
        assert y1 == 824
        # y2 should be box bottom (0.9*1080=972) plus padding (40) = 1012
        assert y2 == 1012

    def test_clamps_to_frame_bounds(self):
        boxes = [BoundingBox(x=0.0, y=0.95, width=1.0, height=0.05)]
        x1, y1, x2, y2 = _compute_crop_region(boxes, 100, 100, padding_px=40)

        assert y1 >= 0
        assert y2 <= 100

    def test_multiple_boxes_covers_all(self):
        boxes = [
            BoundingBox(x=0.1, y=0.7, width=0.3, height=0.05),
            BoundingBox(x=0.5, y=0.9, width=0.3, height=0.05),
        ]
        x1, y1, x2, y2 = _compute_crop_region(boxes, 1000, 1000, padding_px=0)

        # Should span from y=0.7 to y=0.95
        assert y1 == 700
        assert y2 == 950


class TestRemapBoxesToCrop:
    def test_remaps_to_crop_local_coords(self):
        boxes = [BoundingBox(x=0.1, y=0.8, width=0.8, height=0.1)]
        # Crop covers y=800..1000 in a 1000x1000 frame
        remapped = _remap_boxes_to_crop(boxes, 0, 800, 1000, 200, 1000, 1000)

        assert len(remapped) == 1
        box = remapped[0]
        assert abs(box.x - 0.1) < 1e-9
        assert abs(box.y - 0.0) < 1e-9  # 800-800=0 in crop
        assert abs(box.width - 0.8) < 1e-9
        assert abs(box.height - 0.5) < 1e-9  # 100px / 200px crop height


@pytest.fixture
def video_with_subtitle_text(tmp_path):
    """Create a 2-second video with burned-in text at the bottom."""
    ffmpeg = get_ffmpeg()
    output = tmp_path / 'input.mp4'
    cmd = [
        ffmpeg,
        '-y',
        '-f',
        'lavfi',
        '-i',
        'color=size=320x240:duration=2:rate=24:color=darkblue',
        '-f',
        'lavfi',
        '-i',
        'anullsrc=r=44100:cl=stereo',
        '-t',
        '2',
        '-vf',
        "drawtext=text='Test Subtitle':fontsize=20:fontcolor=white:x=(w-tw)/2:y=h-40",
        '-c:v',
        'libx264',
        '-preset',
        'ultrafast',
        '-c:a',
        'aac',
        '-b:a',
        '128k',
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f'Could not create test video: {result.stderr}')
    return output


@pytest.mark.slow
class TestRemoveBurnedInSubtitles:
    def test_produces_output_video(self, video_with_subtitle_text, tmp_path):
        box = BoundingBox(x=0.1, y=0.8, width=0.8, height=0.15)
        ocr_results = [
            OCRResult(timestamp_ms=0, text='Test Subtitle', boxes=[box]),
            OCRResult(timestamp_ms=1000, text='Test Subtitle', boxes=[box]),
        ]
        output = tmp_path / 'output.mp4'

        remove_burned_in_subtitles(
            video_with_subtitle_text,
            output,
            ocr_results,
            device='cpu',
        )

        assert output.exists()
        assert output.stat().st_size > 0

        info = probe_video_encoding(output)
        assert info['width'] == 320
        assert info['height'] == 240
