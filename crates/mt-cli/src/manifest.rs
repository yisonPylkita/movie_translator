//! Persistent download manifest + validation cache.
//!
//! Plain JSON on disk, written atomically (tmp file in same dir + fsync +
//! rename). Corrupt or missing manifest loads as `None` — the user's file is
//! never deleted. SHA-256 is a small pure-Rust implementation (FIPS 180-4) to
//! avoid new dependencies.

use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::{self, Error, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Schema version of the manifest file on disk.
pub const MANIFEST_SCHEMA_VERSION: u32 = 1;
/// Attempt history cap per episode (oldest dropped beyond this).
pub const MAX_ATTEMPTS: usize = 8;

// ── Data model ─────────────────────────────────────────────────────────────

/// Aggregated run summary, derived from episode final statuses.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct Summary {
    pub downloaded: usize,
    pub skipped: usize,
    pub failed: usize,
    pub cancelled: usize,
}

/// Identity of the input JSON the manifest was created from.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct InputIdentity {
    pub title: Option<String>,
    pub source_json_path: Option<PathBuf>,
    pub sha256: Option<String>,
    pub episode_count: usize,
    pub resolved_at: Option<String>,
}

/// Terminal-ish status of one episode in the manifest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum FinalStatus {
    Pending,
    InProgress,
    Complete,
    Failed,
}

/// Status of one mirror attempt.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AttemptStatus {
    Pending,
    Running,
    Ok,
    Failed,
}

/// One attempt at downloading an episode from one mirror.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttemptRecord {
    pub mirror_idx: usize,
    pub host: Option<String>,
    pub url: String,
    pub status: AttemptStatus,
    pub reason: Option<String>,
    pub bytes_downloaded: u64,
    pub secs: f64,
    pub started_at: Option<String>,
}

/// Metadata for a completed output file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OutputMeta {
    pub path: PathBuf,
    pub size: u64,
    pub sha256: Option<String>,
    pub validated: bool,
    pub ffprobe_version: Option<String>,
    pub checked_at: Option<String>,
}

/// One episode in the manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeRecord {
    pub episode: u32,
    pub attempts: Vec<AttemptRecord>,
    pub output: Option<OutputMeta>,
    pub final_status: FinalStatus,
}

/// Validation cache entry, keyed by file path string.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CacheEntry {
    pub size: u64,
    pub mtime_ns: u64,
    pub ok: bool,
    pub reason: Option<String>,
    pub ffprobe_version: Option<String>,
    pub checked_at: Option<String>,
}

/// Root of the persisted manifest.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Manifest {
    pub schema_version: u32,
    pub input: InputIdentity,
    pub episodes: Vec<EpisodeRecord>,
    pub failures: Vec<String>,
    pub summary: Summary,
    pub validation_cache: HashMap<String, CacheEntry>,
}

// ── Manifest API ───────────────────────────────────────────────────────────

impl Manifest {
    pub fn new() -> Self {
        Self {
            schema_version: MANIFEST_SCHEMA_VERSION,
            ..Default::default()
        }
    }

    /// Load a manifest. Missing or corrupt file → `None` (caller warns;
    /// the user's file is left untouched).
    pub fn load(path: &Path) -> Option<Manifest> {
        let bytes = fs::read(path).ok()?;
        serde_json::from_slice::<Manifest>(&bytes).ok()
    }

    /// Atomically persist: write tmp in the same directory, fsync, rename over.
    pub fn save_atomic(&self, path: &Path) -> io::Result<()> {
        let dir = path.parent().unwrap_or_else(|| Path::new("."));
        let file_name = path
            .file_name()
            .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "manifest path lacks file name"))?;
        let tmp = dir.join(format!(".{}.tmp", file_name.to_string_lossy()));
        // Write + fsync the tmp file; any failure removes the tmp so no
        // half-written file survives on disk.
        let write_result = (|| -> io::Result<()> {
            let mut f = File::create(&tmp)?;
            let json = serde_json::to_vec_pretty(self)
                .map_err(|e| Error::new(ErrorKind::InvalidData, e.to_string()))?;
            f.write_all(&json)?;
            f.sync_all()
        })();
        if let Err(e) = write_result {
            let _ = fs::remove_file(&tmp);
            return Err(e);
        }
        if let Err(e) = fs::rename(&tmp, path) {
            let _ = fs::remove_file(&tmp);
            return Err(e);
        }
        // Durability: fsync the parent directory so the rename itself survives
        // a crash (POSIX makes the new entry durable once the dir is synced).
        #[cfg(unix)]
        {
            if let Ok(dir_f) = File::open(dir) {
                let _ = dir_f.sync_all();
            }
        }
        Ok(())
    }

    /// Ensure an episode record exists (created as Pending if absent).
    pub fn ensure_episode(&mut self, ep: u32) {
        if !self.episodes.iter().any(|e| e.episode == ep) {
            self.episodes.push(EpisodeRecord {
                episode: ep,
                attempts: Vec::new(),
                output: None,
                final_status: FinalStatus::Pending,
            });
        }
    }

    /// Record a mirror attempt; history capped at [`MAX_ATTEMPTS`] (oldest dropped).
    pub fn record_attempt(&mut self, ep: u32, attempt: AttemptRecord) {
        if let Some(rec) = self.episodes.iter_mut().find(|e| e.episode == ep) {
            rec.attempts.push(attempt);
            while rec.attempts.len() > MAX_ATTEMPTS {
                rec.attempts.remove(0);
            }
        }
    }

    /// Attach output metadata to an episode.
    pub fn set_output(&mut self, ep: u32, output: OutputMeta) {
        if let Some(rec) = self.episodes.iter_mut().find(|e| e.episode == ep) {
            rec.output = Some(output);
        }
    }

    /// Set an episode's final status.
    pub fn set_final_status(&mut self, ep: u32, status: FinalStatus) {
        if let Some(rec) = self.episodes.iter_mut().find(|e| e.episode == ep) {
            rec.final_status = status;
        }
    }

    /// Validation cache hit: same path, same size AND mtime → cached result.
    /// Any mismatch (or absent entry) is a miss and invalidates the entry.
    pub fn cache_get(&self, path: &str, size: u64, mtime_ns: u64) -> Option<&CacheEntry> {
        match self.validation_cache.get(path) {
            Some(entry) if entry.size == size && entry.mtime_ns == mtime_ns => Some(entry),
            _ => None,
        }
    }

    /// Store a validation result in the cache.
    pub fn cache_put(&mut self, path: String, entry: CacheEntry) {
        self.validation_cache.insert(path, entry);
    }

    /// Reconcile episode statuses against the set of files present on disk:
    /// episodes whose output file vanished are reset to Pending (no output),
    /// so a resumed run re-downloads them.
    pub fn reconcile_episodes(&mut self, present_files: &[PathBuf]) {
        for rec in &mut self.episodes {
            if let Some(out) = &rec.output
                && !present_files.contains(&out.path)
            {
                rec.output = None;
                if rec.final_status == FinalStatus::Complete {
                    rec.final_status = FinalStatus::Pending;
                }
            }
        }
    }

    /// Derive a summary from episode final statuses.
    pub fn to_summary(&self) -> Summary {
        let mut s = Summary::default();
        for rec in &self.episodes {
            match rec.final_status {
                FinalStatus::Complete => s.downloaded += 1,
                FinalStatus::Failed => s.failed += 1,
                FinalStatus::Pending => s.skipped += 1,
                FinalStatus::InProgress => {}
            }
        }
        s
    }
}

// ── SHA-256 (pure Rust, FIPS 180-4) ────────────────────────────────────────

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// Incremental SHA-256 hasher (FIPS 180-4). Feed data with [`Sha256::update`],
/// finish with [`Sha256::finalize`]. Memory-bounded: internal state is a fixed
/// 64-byte block plus 8 words — the caller chooses chunk sizes, so arbitrarily
/// large inputs never need to be buffered whole.
#[derive(Clone)]
pub struct Sha256 {
    h: [u32; 8],
    block: [u8; 64],
    block_len: usize,
    total_len: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    pub fn new() -> Self {
        Self {
            h: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            block: [0u8; 64],
            block_len: 0,
            total_len: 0,
        }
    }

    /// Feed `data` into the hash state.
    pub fn update(&mut self, mut data: &[u8]) {
        self.total_len = self.total_len.wrapping_add(data.len() as u64);
        // Fill a partially-consumed block first.
        if self.block_len > 0 {
            let need = 64 - self.block_len;
            if data.len() >= need {
                self.block[self.block_len..].copy_from_slice(&data[..need]);
                self.compress();
                self.block_len = 0;
                data = &data[need..];
            } else {
                self.block[self.block_len..self.block_len + data.len()].copy_from_slice(data);
                self.block_len += data.len();
                return;
            }
        }
        // Whole blocks straight from the input slice.
        let mut chunks = data.chunks_exact(64);
        for chunk in &mut chunks {
            let mut block = [0u8; 64];
            block.copy_from_slice(chunk);
            self.compress_block(&block);
        }
        // Remainder becomes the next partial block.
        let rem = chunks.remainder();
        if !rem.is_empty() {
            self.block[..rem.len()].copy_from_slice(rem);
            self.block_len = rem.len();
        }
    }

    /// Finish and return the lowercase hex digest.
    pub fn finalize(mut self) -> String {
        let bit_len = self.total_len.wrapping_mul(8);
        self.update(&[0x80]);
        while self.block_len != 56 {
            self.update(&[0]);
        }
        self.update(&bit_len.to_be_bytes());
        self.h.iter().map(|x| format!("{x:08x}")).collect()
    }

    fn compress(&mut self) {
        let mut block = [0u8; 64];
        block.copy_from_slice(&self.block);
        self.compress_block(&block);
    }

    fn compress_block(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for (i, b) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([b[0], b[1], b[2], b[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = self.h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        self.h[0] = self.h[0].wrapping_add(a);
        self.h[1] = self.h[1].wrapping_add(b);
        self.h[2] = self.h[2].wrapping_add(c);
        self.h[3] = self.h[3].wrapping_add(d);
        self.h[4] = self.h[4].wrapping_add(e);
        self.h[5] = self.h[5].wrapping_add(f);
        self.h[6] = self.h[6].wrapping_add(g);
        self.h[7] = self.h[7].wrapping_add(hh);
    }
}

/// SHA-256 digest of `data` as lowercase hex (single-message API; small inputs).
pub fn sha256_hex(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    h.finalize()
}

/// Stream `path` through SHA-256 with a bounded 64 KiB buffer. Memory use is
/// O(64 KiB) regardless of file size.
pub fn sha256_file(path: &Path) -> io::Result<String> {
    let mut f = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize())
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_manifest() -> Manifest {
        let mut m = Manifest::new();
        m.input.title = Some("Test Anime".into());
        m.input.episode_count = 2;
        m.ensure_episode(1);
        m.ensure_episode(2);
        m
    }

    #[test]
    fn manifest_atomic_save_load_roundtrip() {
        let dir = std::env::temp_dir().join("mt-cli-test-manifest-roundtrip");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create dir");
        let path = dir.join("manifest.json");

        let mut m = sample_manifest();
        m.set_final_status(1, FinalStatus::Complete);
        m.set_output(
            1,
            OutputMeta {
                path: PathBuf::from("/tmp/ep1.mkv"),
                size: 1_048_576,
                sha256: Some("abc".into()),
                validated: true,
                ffprobe_version: Some("ffprobe 7.1".into()),
                checked_at: Some("2025-07-30T12:00:00Z".into()),
            },
        );
        m.save_atomic(&path).expect("save");

        let loaded = Manifest::load(&path).expect("load");
        assert_eq!(loaded.schema_version, 1);
        assert_eq!(loaded.input.title.as_deref(), Some("Test Anime"));
        assert_eq!(loaded.episodes.len(), 2);
        let ep1 = loaded
            .episodes
            .iter()
            .find(|e| e.episode == 1)
            .expect("ep 1 present");
        assert_eq!(ep1.final_status, FinalStatus::Complete);
        let out = ep1.output.as_ref().expect("output present");
        assert_eq!(out.size, 1_048_576);
        assert_eq!(out.sha256.as_deref(), Some("abc"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn manifest_missing_returns_none() {
        let path = Path::new("/nonexistent/definitely/missing-manifest.json");
        assert!(Manifest::load(path).is_none());
    }

    #[test]
    fn manifest_corrupt_returns_none() {
        let dir = std::env::temp_dir().join("mt-cli-test-manifest-corrupt");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create dir");
        let path = dir.join("manifest.json");
        fs::write(&path, b"{ not valid json !!!").expect("write corrupt");
        assert!(Manifest::load(&path).is_none(), "corrupt manifest -> None");
        // File must NOT be deleted
        assert!(path.exists(), "corrupt user file preserved");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn manifest_attempts_capped() {
        let mut m = sample_manifest();
        for i in 0..12 {
            m.record_attempt(
                1,
                AttemptRecord {
                    mirror_idx: i,
                    host: Some(format!("host{i}")),
                    url: format!("https://h{i}.example.com/v.mp4"),
                    status: AttemptStatus::Failed,
                    reason: Some("boom".into()),
                    bytes_downloaded: 0,
                    secs: 1.0,
                    started_at: None,
                },
            );
        }
        let ep1 = m.episodes.iter().find(|e| e.episode == 1).expect("ep 1");
        assert_eq!(ep1.attempts.len(), MAX_ATTEMPTS, "history capped at 8");
        assert_eq!(ep1.attempts[0].mirror_idx, 4, "oldest dropped");
        assert_eq!(ep1.attempts[7].mirror_idx, 11, "newest kept");
    }

    #[test]
    fn manifest_resume_preserves_status() {
        let dir = std::env::temp_dir().join("mt-cli-test-manifest-resume");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create dir");
        let path = dir.join("manifest.json");

        let mut m = sample_manifest();
        m.set_final_status(1, FinalStatus::Complete);
        m.set_output(
            1,
            OutputMeta {
                path: dir.join("ep1.mkv"),
                size: 100,
                sha256: None,
                validated: false,
                ffprobe_version: None,
                checked_at: None,
            },
        );
        m.set_final_status(2, FinalStatus::Failed);
        m.save_atomic(&path).expect("save");

        // Episode 1 output exists on disk → status preserved; ep 2 failed → preserved
        let present = vec![dir.join("ep1.mkv")];
        let mut loaded = Manifest::load(&path).expect("load");
        loaded.reconcile_episodes(&present);
        let ep1 = loaded.episodes.iter().find(|e| e.episode == 1).unwrap();
        assert_eq!(ep1.final_status, FinalStatus::Complete);
        assert!(ep1.output.is_some());
        let ep2 = loaded.episodes.iter().find(|e| e.episode == 2).unwrap();
        assert_eq!(ep2.final_status, FinalStatus::Failed);

        // Episode 1 output missing on next resume → reset to Pending
        let mut loaded2 = Manifest::load(&path).expect("load");
        loaded2.reconcile_episodes(&[]);
        let ep1b = loaded2.episodes.iter().find(|e| e.episode == 1).unwrap();
        assert_eq!(ep1b.final_status, FinalStatus::Pending);
        assert!(ep1b.output.is_none());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn manifest_validation_cache_hit_and_mtime_invalidation() {
        let mut m = sample_manifest();
        let key = "/tmp/ep1.mkv".to_string();
        m.cache_put(
            key.clone(),
            CacheEntry {
                size: 1000,
                mtime_ns: 111,
                ok: true,
                reason: None,
                ffprobe_version: None,
                checked_at: Some("t".into()),
            },
        );
        // Same size + mtime → hit
        assert!(m.cache_get(&key, 1000, 111).is_some());
        // Size changed → miss
        assert!(m.cache_get(&key, 2000, 111).is_none());
        // Mtime changed → miss
        assert!(m.cache_get(&key, 1000, 222).is_none());
        // Unknown path → miss
        assert!(m.cache_get("/other", 1000, 111).is_none());
    }

    #[test]
    fn manifest_to_summary_counts() {
        let mut m = sample_manifest();
        m.set_final_status(1, FinalStatus::Complete);
        m.set_final_status(2, FinalStatus::Failed);
        m.ensure_episode(3);
        let s = m.to_summary();
        assert_eq!(s.downloaded, 1);
        assert_eq!(s.failed, 1);
        assert_eq!(s.skipped, 1);
    }

    #[test]
    fn sha256_known_vectors() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"hello world"),
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    #[test]
    fn sha256_incremental_chunked_matches_one_shot() {
        // Odd chunk sizes exercise partial-block fill, exact-block, and
        // remainder paths of the incremental hasher.
        let data: Vec<u8> = (0..4096u32)
            .map(|i| (i.wrapping_mul(31) & 0xff) as u8)
            .collect();
        let one_shot = sha256_hex(&data);
        let mut h = Sha256::new();
        for chunk in data.chunks(7) {
            h.update(chunk);
        }
        assert_eq!(h.finalize(), one_shot);

        // Also feed one byte at a time.
        let mut h2 = Sha256::new();
        for b in &data {
            h2.update(&[*b]);
        }
        assert_eq!(h2.finalize(), one_shot);
    }

    #[test]
    fn engine_sha256_streams_large_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("big.bin");
        // ~3 MiB + 37 bytes of deterministic pseudo-random data: many block
        // boundaries, a non-64-aligned tail, and more than a 64 KiB buffer.
        let mut data = Vec::with_capacity(3 * 1024 * 1024 + 37);
        let mut x = 0x1234_5678u64;
        while data.len() < 3 * 1024 * 1024 + 37 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            data.extend_from_slice(&x.to_le_bytes());
        }
        fs::write(&path, &data).expect("write big file");
        let expected = sha256_hex(&data);
        let streamed = sha256_file(&path).expect("stream hash");
        assert_eq!(streamed, expected, "streamed digest must match single-shot");
    }

    #[test]
    fn manifest_save_fsync_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("manifest.json");
        let m = sample_manifest();
        m.save_atomic(&path).expect("save with dir fsync");
        assert!(path.exists(), "manifest written");
        assert!(Manifest::load(&path).is_some(), "round-trips");
        // No tmp left after a successful save.
        assert!(
            !dir.path().join(".manifest.json.tmp").exists(),
            "no tmp after success"
        );
    }

    #[test]
    fn manifest_tmp_cleaned_on_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("manifest.json");
        let m = sample_manifest();
        m.save_atomic(&path).expect("initial save");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            // Make the directory read-only so the tmp write fails mid-save.
            let perms = fs::metadata(dir.path()).expect("dir meta").permissions();
            let mut ro = perms.clone();
            ro.set_mode(0o500);
            fs::set_permissions(dir.path(), ro).expect("make read-only");

            let err = m.save_atomic(&path);
            assert!(err.is_err(), "save must fail on read-only dir");
            assert!(
                !dir.path().join(".manifest.json.tmp").exists(),
                "no leftover tmp on write error"
            );

            fs::set_permissions(dir.path(), perms).expect("restore perms");
        }
    }

    #[test]
    fn engine_atomic_interruption_manifest_valid() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("manifest.json");

        let mut m = sample_manifest();
        m.set_final_status(1, FinalStatus::Complete);
        m.save_atomic(&path).expect("save");

        // Simulate a crash mid-save: a garbage tmp file next to the valid
        // manifest. Load must still return the prior valid manifest.
        fs::write(dir.path().join(".manifest.json.tmp"), b"{ corrupted !!!")
            .expect("write garbage tmp");

        let loaded = Manifest::load(&path).expect("prior valid manifest loads");
        assert_eq!(loaded.episodes.len(), 2);
        let ep1 = loaded.episodes.iter().find(|e| e.episode == 1).unwrap();
        assert_eq!(ep1.final_status, FinalStatus::Complete);
    }
}
