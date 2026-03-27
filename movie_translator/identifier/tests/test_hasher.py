import struct

import pytest

from movie_translator.identifier.hasher import compute_oshash


class TestComputeOshash:
    def test_computes_hash_for_small_file(self, tmp_path):
        """A file smaller than 128KB: entire file is both 'first 64KB' and 'last 64KB'."""
        f = tmp_path / 'small.bin'
        # Write 16 bytes: two uint64 values (1 and 2)
        data = struct.pack('<QQ', 1, 2)
        f.write_bytes(data)
        result = compute_oshash(f)
        # hash = filesize + sum_of_chunks
        # filesize=16, first_64k covers whole file, last_64k overlaps
        # For files < 128KB, read first min(64KB, filesize) and last min(64KB, filesize)
        # Both reads cover the same data, so chunks are summed twice
        # sum = (1+2)*2 + 16 = 22
        expected = format(22, '016x')
        assert result == expected
        assert len(result) == 16

    def test_returns_16_char_hex_string(self, tmp_path):
        f = tmp_path / 'zeros.bin'
        f.write_bytes(b'\x00' * 256)
        result = compute_oshash(f)
        assert len(result) == 16
        int(result, 16)  # Should not raise — valid hex

    def test_hash_changes_with_content(self, tmp_path):
        f1 = tmp_path / 'a.bin'
        f2 = tmp_path / 'b.bin'
        f1.write_bytes(b'\x01' * 1024)
        f2.write_bytes(b'\x02' * 1024)
        assert compute_oshash(f1) != compute_oshash(f2)

    def test_empty_file_raises(self, tmp_path):
        f = tmp_path / 'empty.bin'
        f.write_bytes(b'')
        with pytest.raises(ValueError, match='empty'):
            compute_oshash(f)
