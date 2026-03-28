import hashlib

import pytest

from movie_translator.identifier.napihash import compute_napiprojekt_hash


class TestNapiProjektHash:
    def test_hash_of_small_file(self, tmp_path):
        """File smaller than 10MB: hash entire content."""
        f = tmp_path / 'small.mkv'
        f.write_bytes(b'hello world')
        result = compute_napiprojekt_hash(f)
        # MD5 of b'hello world'
        assert result == '5eb63bbbe01eeed093cb22bb8f5acdc3'

    def test_hash_of_large_file_reads_only_10mb(self, tmp_path):
        """File larger than 10MB: hash only first 10MB."""
        f = tmp_path / 'large.mkv'
        chunk = b'\x00' * (10 * 1024 * 1024)  # exactly 10MB of zeros
        extra = b'\xff' * 1024  # extra data beyond 10MB
        f.write_bytes(chunk + extra)
        result = compute_napiprojekt_hash(f)
        expected = hashlib.md5(chunk).hexdigest()
        assert result == expected

    def test_empty_file_raises(self, tmp_path):
        f = tmp_path / 'empty.mkv'
        f.write_bytes(b'')
        with pytest.raises(ValueError):
            compute_napiprojekt_hash(f)
