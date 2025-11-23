#!/usr/bin/env python3
"""
Subtitle validation script for movie_translator.

Validates that English and Polish subtitle files match in:
- Number of subtitle entries
- Timing (start/end times)
- Format structure
- HTML/formatting tags presence
"""

import sys
from pathlib import Path
from typing import Tuple, List
import pysubs2
from pysubs2 import SSAFile, SSAEvent


class SubtitleValidator:
    """Validates matching between source and translated subtitle files."""

    def __init__(self, tolerance_ms: int = 50):
        """
        Initialize validator.

        Args:
            tolerance_ms: Allowed timing difference in milliseconds (default 50ms)
        """
        self.tolerance_ms = tolerance_ms
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self, source_path: Path, target_path: Path) -> bool:
        """
        Validate that two subtitle files match in structure and timing.

        Args:
            source_path: Path to source (English) subtitle file
            target_path: Path to target (Polish) subtitle file

        Returns:
            True if validation passes, False otherwise
        """
        self.errors.clear()
        self.warnings.clear()

        # Load subtitle files
        try:
            source_subs = pysubs2.load(str(source_path))
        except Exception as e:
            self.errors.append(f"Failed to load source file '{source_path}': {e}")
            return False

        try:
            target_subs = pysubs2.load(str(target_path))
        except Exception as e:
            self.errors.append(f"Failed to load target file '{target_path}': {e}")
            return False

        # Validate entry count
        if len(source_subs) != len(target_subs):
            self.errors.append(
                f"Entry count mismatch: source has {len(source_subs)} entries, "
                f"target has {len(target_subs)} entries"
            )
            return False

        # Validate each subtitle entry
        for idx, (source_event, target_event) in enumerate(
            zip(source_subs, target_subs), start=1
        ):
            self._validate_event(idx, source_event, target_event)

        return len(self.errors) == 0

    def _validate_event(
        self, index: int, source: SSAEvent, target: SSAEvent
    ) -> None:
        """
        Validate a single subtitle event pair.

        Args:
            index: Subtitle index (1-based for display)
            source: Source subtitle event
            target: Target subtitle event
        """
        # Check start time
        start_diff = abs(source.start - target.start)
        if start_diff > self.tolerance_ms:
            self.errors.append(
                f"Entry {index}: Start time mismatch - "
                f"source: {self._format_time(source.start)}, "
                f"target: {self._format_time(target.start)} "
                f"(diff: {start_diff}ms)"
            )

        # Check end time
        end_diff = abs(source.end - target.end)
        if end_diff > self.tolerance_ms:
            self.errors.append(
                f"Entry {index}: End time mismatch - "
                f"source: {self._format_time(source.end)}, "
                f"target: {self._format_time(target.end)} "
                f"(diff: {end_diff}ms)"
            )

        # Check duration consistency
        source_duration = source.end - source.start
        target_duration = target.end - target.start
        duration_diff = abs(source_duration - target_duration)
        if duration_diff > self.tolerance_ms:
            self.errors.append(
                f"Entry {index}: Duration mismatch - "
                f"source: {source_duration}ms, target: {target_duration}ms "
                f"(diff: {duration_diff}ms)"
            )

        # Check for HTML/formatting tags consistency
        self._validate_formatting_tags(index, source.text, target.text)

    def _validate_formatting_tags(
        self, index: int, source_text: str, target_text: str
    ) -> None:
        """
        Check that HTML/formatting tags are preserved in translation.

        Args:
            index: Subtitle index
            source_text: Source text
            target_text: Target text
        """
        # Common subtitle formatting tags
        tags = ["<i>", "</i>", "<b>", "</b>", "<u>", "</u>"]

        for tag in tags:
            source_count = source_text.count(tag)
            target_count = target_text.count(tag)

            if source_count != target_count:
                self.warnings.append(
                    f"Entry {index}: Tag '{tag}' count mismatch - "
                    f"source: {source_count}, target: {target_count}"
                )

        # Check for newlines (multiline subtitles)
        source_lines = source_text.count("\\N")
        target_lines = target_text.count("\\N")

        if source_lines != target_lines:
            self.warnings.append(
                f"Entry {index}: Line break count mismatch - "
                f"source: {source_lines}, target: {target_lines}"
            )

    @staticmethod
    def _format_time(milliseconds: int) -> str:
        """Format milliseconds as HH:MM:SS,mmm."""
        hours = milliseconds // 3600000
        minutes = (milliseconds % 3600000) // 60000
        seconds = (milliseconds % 60000) // 1000
        ms = milliseconds % 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"

    def print_results(self) -> None:
        """Print validation results."""
        if self.errors:
            print("\n❌ VALIDATION FAILED\n")
            print(f"Found {len(self.errors)} error(s):\n")
            for error in self.errors:
                print(f"  ❌ {error}")
        else:
            print("\n✅ VALIDATION PASSED")
            print("  Subtitle files match in structure and timing.")

        if self.warnings:
            print(f"\n⚠️  Found {len(self.warnings)} warning(s):\n")
            for warning in self.warnings:
                print(f"  ⚠️  {warning}")

        print()


def main():
    """Main entry point."""
    if len(sys.argv) != 3:
        print("Usage: python validate_subtitles.py <source_srt> <target_srt>")
        print("\nExample:")
        print("  python validate_subtitles.py subtitles/en_full.srt subtitles/pl_full.srt")
        sys.exit(1)

    source_path = Path(sys.argv[1])
    target_path = Path(sys.argv[2])

    # Check files exist
    if not source_path.exists():
        print(f"❌ Error: Source file not found: {source_path}")
        sys.exit(1)

    if not target_path.exists():
        print(f"❌ Error: Target file not found: {target_path}")
        sys.exit(1)

    # Validate
    print(f"Validating subtitles:")
    print(f"  Source: {source_path}")
    print(f"  Target: {target_path}")

    validator = SubtitleValidator(tolerance_ms=50)
    success = validator.validate(source_path, target_path)
    validator.print_results()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
