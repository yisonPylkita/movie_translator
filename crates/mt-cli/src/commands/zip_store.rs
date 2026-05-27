//! Store-only (uncompressed) ZIP packing for the `iphone --zip` flow.
//!
//! Port of `movie_translator/iphone/zip_packer.py` (`pack_and_clean` +
//! `verify_zip`). iPhones happily mount an uncompressed ZIP and play the MP4s
//! straight out of it, so we use `CompressionMethod::Stored` — no transcode,
//! no compression, just a container.
//!
//! Built on the `zip` crate rather than a hand-rolled implementation so that:
//!   * 64-bit sizes are handled (the crate emits Zip64 records automatically),
//!     so files >4 GB don't silently truncate;
//!   * files are *streamed* into the archive via [`std::io::copy`] — never read
//!     fully into a `Vec<u8>` — so memory stays bounded on episode-sized MP4s;
//!   * reading/verifying existing archives goes through [`zip::ZipArchive`],
//!     which returns `Result` (no out-of-bounds indexing panics) and validates
//!     each entry's CRC-32 on read.

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

/// Pack `mp4_files` into `<input_dir>.zip` (store-only), then delete the
/// packed source files. Port of `pack_and_clean`.
///
/// Resumable/idempotent: if the target zip already exists, its existing entries
/// are carried over and only new arcnames are appended. The write goes to a
/// `.partial` sibling that is verified (CRC-checked) and then atomically renamed
/// over the target, so a crashed run never leaves a half-written archive in
/// place. On any error the `.partial` is removed and the sources are kept.
pub fn pack_and_clean(input_dir: &Path, mp4_files: &[PathBuf]) -> Result<PathBuf> {
    let name = input_dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    let parent = input_dir.parent().unwrap_or(Path::new("."));
    let zip_path = parent.join(format!("{name}.zip"));
    let partial = parent.join(format!("{name}.zip.partial"));

    let build = || -> Result<()> {
        // Carry over existing entries for resume/idempotency.
        let existing: HashSet<String> = if zip_path.exists() {
            read_zip_arcnames(&zip_path)
                .with_context(|| format!("reading existing zip {}", zip_path.display()))?
        } else {
            HashSet::new()
        };

        write_store_zip(&partial, &zip_path, input_dir, mp4_files, &existing)
            .with_context(|| format!("writing zip {}", partial.display()))?;
        verify_zip(&partial).with_context(|| format!("verifying zip {}", partial.display()))?;
        std::fs::rename(&partial, &zip_path)
            .with_context(|| format!("renaming {} -> {}", partial.display(), zip_path.display()))?;
        Ok(())
    };

    if let Err(e) = build() {
        if partial.exists() {
            let _ = std::fs::remove_file(&partial);
        }
        return Err(e);
    }

    // Sources are deleted only after a verified, atomically-installed archive.
    for f in mp4_files {
        let _ = std::fs::remove_file(f);
    }
    Ok(zip_path)
}

/// Write a fresh store-only zip at `partial`, containing the union of any
/// `existing` entries (copied from `source_zip` if it exists) and the new
/// `files` (relative to `base_dir`), skipping arcnames already in `existing`.
///
/// Each file is streamed in with [`io::copy`]; nothing is buffered whole.
fn write_store_zip(
    partial: &Path,
    source_zip: &Path,
    base_dir: &Path,
    files: &[PathBuf],
    existing: &HashSet<String>,
) -> Result<()> {
    let out = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(partial)
        .with_context(|| format!("creating {}", partial.display()))?;
    let mut writer = ZipWriter::new(out);
    let options = SimpleFileOptions::default()
        .compression_method(CompressionMethod::Stored)
        // Let the crate emit Zip64 records when an entry/offset exceeds 4 GB.
        .large_file(true);

    // Copy over existing entries first (resume case), streaming each.
    if source_zip.exists() {
        let f = File::open(source_zip)
            .with_context(|| format!("opening existing zip {}", source_zip.display()))?;
        let mut archive = ZipArchive::new(io::BufReader::new(f))
            .with_context(|| format!("parsing existing zip {}", source_zip.display()))?;
        for i in 0..archive.len() {
            let mut entry = archive
                .by_index(i)
                .with_context(|| format!("reading entry {i} of {}", source_zip.display()))?;
            let entry_name = entry.name().to_string();
            writer
                .start_file(&entry_name, options)
                .with_context(|| format!("starting carried-over entry {entry_name}"))?;
            io::copy(&mut entry, &mut writer)
                .with_context(|| format!("copying carried-over entry {entry_name}"))?;
        }
    }

    for f in files {
        let arcname = f
            .strip_prefix(base_dir)
            .map(|p| p.to_string_lossy().replace('\\', "/"))
            .map_err(|_| anyhow::anyhow!("{} is not under {}", f.display(), base_dir.display()))?;
        if existing.contains(&arcname) {
            continue;
        }
        let mut src = File::open(f).with_context(|| format!("opening {}", f.display()))?;
        writer
            .start_file(&arcname, options)
            .with_context(|| format!("starting entry {arcname}"))?;
        io::copy(&mut src, &mut writer)
            .with_context(|| format!("streaming {} into archive", f.display()))?;
    }

    writer
        .finish()
        .context("finalising zip central directory")?;
    Ok(())
}

/// Read the set of arcnames in `zip_path`. Returns an `Err` (never panics) on a
/// truncated/garbage/non-ZIP file.
pub fn read_zip_arcnames(zip_path: &Path) -> Result<HashSet<String>> {
    let f = File::open(zip_path).with_context(|| format!("opening {}", zip_path.display()))?;
    let mut archive = ZipArchive::new(io::BufReader::new(f))
        .with_context(|| format!("parsing zip {}", zip_path.display()))?;
    let mut names = HashSet::with_capacity(archive.len());
    for i in 0..archive.len() {
        let entry = archive
            .by_index(i)
            .with_context(|| format!("reading entry {i}"))?;
        names.insert(entry.name().to_string());
    }
    Ok(names)
}

/// Verify every entry against its stored CRC-32, mirroring Python's
/// `zip_packer.verify_zip` / `zipfile.testzip()`. The `zip` crate validates the
/// CRC while reading an entry to its end, so we stream each entry to a sink.
/// Returns an `Err` (never panics) on a corrupt entry or a malformed archive.
pub fn verify_zip(zip_path: &Path) -> Result<()> {
    let f = File::open(zip_path).with_context(|| format!("opening {}", zip_path.display()))?;
    let mut archive = ZipArchive::new(io::BufReader::new(f))
        .with_context(|| format!("parsing zip {}", zip_path.display()))?;
    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .with_context(|| format!("reading entry {i}"))?;
        let name = entry.name().to_string();
        // Draining the entry forces the crate's CRC-32 check; a mismatch surfaces
        // as an `io::Error` here rather than silent corruption.
        io::copy(&mut entry, &mut io::sink())
            .with_context(|| format!("CRC verification failed for entry {name}"))?;
    }
    Ok(())
}

/// Read every entry's (name, bytes). Test helper / used where small in-memory
/// payloads are expected (round-trip assertions). Returns `Err` on a bad zip.
#[cfg(test)]
pub fn read_zip_entries(zip_path: &Path) -> Result<Vec<(String, Vec<u8>)>> {
    use std::io::Read;
    let f = File::open(zip_path)?;
    let mut archive = ZipArchive::new(io::BufReader::new(f))?;
    let mut entries = Vec::with_capacity(archive.len());
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let name = entry.name().to_string();
        let mut data = Vec::new();
        entry.read_to_end(&mut data)?;
        entries.push((name, data));
    }
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zip_round_trips_store_only() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path();
        let f1 = base.join("a.mp4");
        let f2 = base.join("sub/b.mp4");
        std::fs::create_dir_all(base.join("sub")).unwrap();
        std::fs::write(&f1, b"hello").unwrap();
        std::fs::write(&f2, b"world!!").unwrap();

        let zip = pack_and_clean(base, &[f1.clone(), f2.clone()]).unwrap();
        assert!(zip.exists());
        // source files deleted after packing
        assert!(!f1.exists() && !f2.exists());

        let entries = read_zip_entries(&zip).unwrap();
        let names: HashSet<_> = entries.iter().map(|(n, _)| n.clone()).collect();
        assert!(names.contains("a.mp4"));
        assert!(names.contains("sub/b.mp4"));
        for (n, data) in entries {
            if n == "a.mp4" {
                assert_eq!(data, b"hello");
            } else if n == "sub/b.mp4" {
                assert_eq!(data, b"world!!");
            }
        }
    }

    #[test]
    fn zip_append_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path();
        let f1 = base.join("a.mp4");
        std::fs::write(&f1, b"first").unwrap();
        pack_and_clean(base, std::slice::from_ref(&f1)).unwrap();

        // Second file, append to existing zip.
        let f2 = base.join("c.mp4");
        std::fs::write(&f2, b"second").unwrap();
        let zip = pack_and_clean(base, std::slice::from_ref(&f2)).unwrap();

        let names = read_zip_arcnames(&zip).unwrap();
        assert!(names.contains("a.mp4"));
        assert!(names.contains("c.mp4"));

        // Re-packing an already-present arcname is a no-op (idempotent): pack a
        // file that re-creates "a.mp4" and confirm the original content stays.
        let f1_again = base.join("a.mp4");
        std::fs::write(&f1_again, b"DIFFERENT").unwrap();
        let zip = pack_and_clean(base, std::slice::from_ref(&f1_again)).unwrap();
        let entries = read_zip_entries(&zip).unwrap();
        let a = entries.iter().find(|(n, _)| n == "a.mp4").unwrap();
        assert_eq!(a.1, b"first", "existing arcname must not be overwritten");
    }

    #[test]
    fn verify_zip_detects_corruption() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path();
        let f1 = base.join("a.mp4");
        std::fs::write(&f1, b"hello world payload").unwrap();
        let zip = pack_and_clean(base, std::slice::from_ref(&f1)).unwrap();

        // A freshly written zip verifies cleanly.
        verify_zip(&zip).expect("pristine zip must verify");

        // Corrupt a byte inside the stored (uncompressed) payload; the CRC check
        // on read must now fail.
        let mut bytes = std::fs::read(&zip).unwrap();
        let needle = b"hello";
        let idx = bytes
            .windows(needle.len())
            .position(|w| w == needle)
            .expect("payload present");
        bytes[idx] ^= 0xFF;
        std::fs::write(&zip, &bytes).unwrap();

        assert!(
            verify_zip(&zip).is_err(),
            "corrupted zip must fail verification"
        );
    }

    #[test]
    fn truncated_or_garbage_zip_errors_not_panics() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path();

        // Garbage / non-ZIP file.
        let garbage = base.join("garbage.zip");
        std::fs::write(&garbage, b"this is definitely not a zip file").unwrap();
        assert!(read_zip_arcnames(&garbage).is_err());
        assert!(verify_zip(&garbage).is_err());

        // Truncated ZIP: write a real one then chop it in half.
        let f1 = base.join("a.mp4");
        std::fs::write(&f1, b"some payload bytes here").unwrap();
        let zip = pack_and_clean(base, std::slice::from_ref(&f1)).unwrap();
        let bytes = std::fs::read(&zip).unwrap();
        let truncated = base.join("truncated.zip");
        std::fs::write(&truncated, &bytes[..bytes.len() / 2]).unwrap();
        assert!(read_zip_arcnames(&truncated).is_err());
        assert!(verify_zip(&truncated).is_err());
    }

    #[test]
    fn empty_zip_file_errors() {
        let dir = tempfile::tempdir().unwrap();
        let empty = dir.path().join("empty.zip");
        std::fs::write(&empty, b"").unwrap();
        assert!(read_zip_arcnames(&empty).is_err());
    }
}
