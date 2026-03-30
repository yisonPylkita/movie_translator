"""Tests for AppleTranslationBackend.

Platform-dependent tests (require macOS 26+ with languages installed)
are marked with @pytest.mark.apple_translation and skipped elsewhere.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from movie_translator.translation.apple_backend import (
    AppleTranslationBackend,
    AppleTranslationError,
    _apply_fallbacks,
    _call_swift_binary,
    _ensure_binary,
    is_available,
)

# Skip integration tests if not on macOS 26+
apple_translation = pytest.mark.skipif(
    not is_available(),
    reason='Requires macOS 26+ with Apple Translation',
)


class TestIsAvailable:
    def test_returns_false_on_linux(self):
        with patch('movie_translator.translation.apple_backend.platform') as mock_plat:
            mock_plat.system.return_value = 'Linux'
            assert is_available() is False

    def test_returns_false_on_old_macos(self):
        with patch('movie_translator.translation.apple_backend.platform') as mock_plat:
            mock_plat.system.return_value = 'Darwin'
            mock_plat.mac_ver.return_value = ('15.4', ('', '', ''), '')
            assert is_available() is False

    def test_returns_false_when_source_missing(self):
        with (
            patch('movie_translator.translation.apple_backend.platform') as mock_plat,
            patch.object(Path, 'exists', return_value=False),
        ):
            mock_plat.system.return_value = 'Darwin'
            mock_plat.mac_ver.return_value = ('26.4', ('', '', ''), '')
            assert is_available() is False


class TestCallSwiftBinary:
    def test_parses_success_response(self):
        mock_result = MagicMock()
        mock_result.stdout = json.dumps(
            {
                'translations': ['Cześć', 'Do widzenia'],
                'elapsed_ms': 100,
            }
        ).encode()
        mock_result.returncode = 0

        with patch(
            'movie_translator.translation.apple_backend.subprocess.run', return_value=mock_result
        ):
            result = _call_swift_binary(Path('/fake/binary'), ['Hello', 'Goodbye'])

        assert result == ['Cześć', 'Do widzenia']

    def test_raises_on_error_response(self):
        mock_result = MagicMock()
        mock_result.stdout = json.dumps(
            {
                'error': 'Languages not installed',
                'code': 'not_installed',
            }
        ).encode()
        mock_result.returncode = 1

        with patch(
            'movie_translator.translation.apple_backend.subprocess.run', return_value=mock_result
        ):
            with pytest.raises(AppleTranslationError, match='not installed'):
                _call_swift_binary(Path('/fake/binary'), ['Hello'])

    def test_raises_on_empty_output(self):
        mock_result = MagicMock()
        mock_result.stdout = b''
        mock_result.stderr = b'crash'
        mock_result.returncode = 1

        with patch(
            'movie_translator.translation.apple_backend.subprocess.run', return_value=mock_result
        ):
            with pytest.raises(AppleTranslationError, match='no output'):
                _call_swift_binary(Path('/fake/binary'), ['Hello'])

    def test_raises_on_invalid_json(self):
        mock_result = MagicMock()
        mock_result.stdout = b'not json at all'
        mock_result.returncode = 0

        with patch(
            'movie_translator.translation.apple_backend.subprocess.run', return_value=mock_result
        ):
            with pytest.raises(AppleTranslationError, match='Invalid JSON'):
                _call_swift_binary(Path('/fake/binary'), ['Hello'])

    def test_raises_on_count_mismatch(self):
        mock_result = MagicMock()
        mock_result.stdout = json.dumps(
            {
                'translations': ['Cześć'],  # sent 2, got 1
                'elapsed_ms': 50,
            }
        ).encode()
        mock_result.returncode = 0

        with patch(
            'movie_translator.translation.apple_backend.subprocess.run', return_value=mock_result
        ):
            with pytest.raises(AppleTranslationError, match='count mismatch'):
                _call_swift_binary(Path('/fake/binary'), ['Hello', 'Goodbye'])


class TestApplyFallbacks:
    def test_empty_translation_uses_original(self):
        result = _apply_fallbacks(
            originals=['Hello world'],
            translations=[''],
            skip_indices=set(),
            cached_translations={},
        )
        assert result == ['Hello world']

    def test_suspiciously_short_uses_original(self):
        result = _apply_fallbacks(
            originals=['What a beautiful day'],
            translations=['X'],
            skip_indices=set(),
            cached_translations={},
        )
        assert result == ['What a beautiful day']

    def test_valid_short_translation_kept(self):
        result = _apply_fallbacks(
            originals=['Yes'],
            translations=['Tak'],
            skip_indices=set(),
            cached_translations={},
        )
        assert result == ['Tak']

    def test_cached_translation_used(self):
        result = _apply_fallbacks(
            originals=['Thank you'],
            translations=['ignored'],
            skip_indices={0},
            cached_translations={0: 'Dziękuję'},
        )
        assert result == ['Dziękuję']

    def test_normal_translation_passes_through(self):
        result = _apply_fallbacks(
            originals=['Hello world', 'Goodbye'],
            translations=['Witaj świecie', 'Do widzenia'],
            skip_indices=set(),
            cached_translations={},
        )
        assert result == ['Witaj świecie', 'Do widzenia']


class TestEnsureBinary:
    def test_compiles_when_missing(self, tmp_path):
        source = tmp_path / 'translate_bridge.swift'
        source.write_text('// test')
        binary = tmp_path / 'translate_bridge'
        # binary does NOT exist yet — should trigger compilation

        with (
            patch('movie_translator.translation.apple_backend._SWIFT_SOURCE', source),
            patch('movie_translator.translation.apple_backend._SWIFT_BINARY', binary),
            patch(
                'movie_translator.translation.apple_backend.shutil.which',
                return_value='/usr/bin/swiftc',
            ),
            patch('movie_translator.translation.apple_backend.subprocess.run') as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stderr='')
            _ensure_binary()
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert '-framework' in call_args
            assert 'Translation' in call_args

    def test_skips_compile_when_binary_newer(self, tmp_path):
        source = tmp_path / 'translate_bridge.swift'
        source.write_text('// test')
        binary = tmp_path / 'translate_bridge'
        binary.touch()
        # Make binary newer than source
        os.utime(binary, (source.stat().st_mtime + 10, source.stat().st_mtime + 10))

        with (
            patch('movie_translator.translation.apple_backend._SWIFT_SOURCE', source),
            patch('movie_translator.translation.apple_backend._SWIFT_BINARY', binary),
            patch('movie_translator.translation.apple_backend.subprocess.run') as mock_run,
        ):
            _ensure_binary()
            mock_run.assert_not_called()

    def test_raises_when_source_missing(self, tmp_path):
        source = tmp_path / 'nonexistent.swift'
        with patch('movie_translator.translation.apple_backend._SWIFT_SOURCE', source):
            with pytest.raises(FileNotFoundError, match='Swift bridge source not found'):
                _ensure_binary()

    def test_raises_when_swiftc_missing(self, tmp_path):
        source = tmp_path / 'translate_bridge.swift'
        source.write_text('// test')
        binary = tmp_path / 'translate_bridge'

        with (
            patch('movie_translator.translation.apple_backend._SWIFT_SOURCE', source),
            patch('movie_translator.translation.apple_backend._SWIFT_BINARY', binary),
            patch('movie_translator.translation.apple_backend.shutil.which', return_value=None),
        ):
            with pytest.raises(FileNotFoundError, match='swiftc.*not found'):
                _ensure_binary()


class TestAppleTranslationBackend:
    def _mock_swift_response(self, translations):
        """Helper to create a mock subprocess result."""
        mock_result = MagicMock()
        mock_result.stdout = json.dumps(
            {
                'translations': translations,
                'elapsed_ms': 100,
            }
        ).encode()
        mock_result.returncode = 0
        return mock_result

    def test_translate_texts_with_mocked_binary(self):
        # "Fine, thank you." is <=3 words so it gets a solo group.
        # The merger sends 2 texts to the binary.
        mock_result = self._mock_swift_response(['Cześć, jak się masz?', 'Dobrze, dziękuję.'])

        with (
            patch(
                'movie_translator.translation.apple_backend._ensure_binary',
                return_value=Path('/fake'),
            ),
            patch(
                'movie_translator.translation.apple_backend.subprocess.run',
                return_value=mock_result,
            ),
        ):
            backend = AppleTranslationBackend(batch_size=200, enable_enhancements=False)
            result = backend.translate_texts(['Hello, how are you?', 'Fine, thank you.'])

        assert len(result) == 2
        assert result[0]  # non-empty
        assert result[1]  # non-empty

    def test_empty_input_returns_empty(self):
        with patch(
            'movie_translator.translation.apple_backend._ensure_binary', return_value=Path('/fake')
        ):
            backend = AppleTranslationBackend(enable_enhancements=False)
            assert backend.translate_texts([]) == []

    def test_progress_callback_called(self):
        mock_result = self._mock_swift_response(['Cześć'])
        callback = MagicMock()

        with (
            patch(
                'movie_translator.translation.apple_backend._ensure_binary',
                return_value=Path('/fake'),
            ),
            patch(
                'movie_translator.translation.apple_backend.subprocess.run',
                return_value=mock_result,
            ),
        ):
            backend = AppleTranslationBackend(enable_enhancements=False)
            backend.translate_texts(['Hello'], progress_callback=callback)

        callback.assert_called_once()
        lines_done, total_lines, rate = callback.call_args[0]
        assert lines_done > 0
        assert total_lines == 1

    def test_cleanup_is_noop(self):
        with patch(
            'movie_translator.translation.apple_backend._ensure_binary', return_value=Path('/fake')
        ):
            backend = AppleTranslationBackend(enable_enhancements=False)
            backend.cleanup()  # should not raise


@apple_translation
class TestAppleTranslationIntegration:
    """Integration tests — only run on macOS 26+ with languages installed."""

    def test_single_translation(self):
        backend = AppleTranslationBackend(enable_enhancements=False)
        result = backend.translate_texts(['Hello world'])
        assert len(result) == 1
        assert result[0]
        assert result[0] != 'Hello world'

    def test_batch_translation(self):
        backend = AppleTranslationBackend(enable_enhancements=False)
        texts = ['What on earth?!', 'Look over there!', 'Guess I will go.']
        result = backend.translate_texts(texts)
        assert len(result) == 3
        assert all(t for t in result)

    def test_with_enhancements(self):
        backend = AppleTranslationBackend(enable_enhancements=True)
        result = backend.translate_texts(['Yes', 'Thank you', 'What a beautiful day!'])
        assert len(result) == 3

    def test_sentence_merging_round_trip(self):
        """Verify that fragment merging + unmerging preserves line count."""
        backend = AppleTranslationBackend(enable_enhancements=False)
        texts = [
            'The monster is wreaking',
            'destruction on an unprecedented scale!',
            'Get me that threat level assessment!',
        ]
        result = backend.translate_texts(texts)
        assert len(result) == 3
        assert all(t for t in result)
