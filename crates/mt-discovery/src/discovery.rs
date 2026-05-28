//! Recursive video file discovery and working directory creation.

use mt_core::Result;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Video file extensions considered valid inputs.
const VIDEO_EXTENSIONS: &[&str] = &["mkv", "mp4"];

/// Infix that marks in-place mux temp files left by a crashed run.
/// E.g. `Episode01.translating.mkv` has stem `Episode01.translating`.
const IN_PLACE_TEMP_INFIX: &str = ".translating";

/// Returns `true` if `path` looks like an in-place mux temp file.
///
/// Checks whether the stem (everything except the final extension) ends with
/// `.translating`.
fn is_in_place_temp(path: &Path) -> bool {
    path.file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.ends_with(IN_PLACE_TEMP_INFIX))
        .unwrap_or(false)
}

/// Returns `true` if `path` has a recognised video extension (case-insensitive).
fn has_video_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| VIDEO_EXTENSIONS.contains(&e.to_lowercase().as_str()))
        .unwrap_or(false)
}

/// Find all video files recursively from any input path.
///
/// - If `input_path` is a file: returns `[input_path]` if it's a video, else `[]`.
/// - If a directory: recursively finds all `.mkv`/`.mp4` files, sorted.
/// - Skips hidden directories (any component starting with `'.'`).
/// - Skips in-place mux temp files (`*.translating.<ext>`).
/// - Returns `[]` for nonexistent paths.
pub fn find_videos(input_path: &Path) -> Vec<PathBuf> {
    if !input_path.exists() {
        return Vec::new();
    }

    if input_path.is_file() {
        if has_video_extension(input_path) && !is_in_place_temp(input_path) {
            return vec![input_path.to_path_buf()];
        }
        return Vec::new();
    }

    let mut videos: Vec<PathBuf> = Vec::new();
    let mut visited: HashSet<PathBuf> = HashSet::new();
    collect_videos(input_path, input_path, &mut videos, &mut visited);
    videos.sort();
    videos
}

/// Recursively walk a directory, collecting video files into `out`.
///
/// Guards against infinite recursion on circular directory symlinks by
/// tracking canonicalized directory paths in `visited` and by never
/// descending into a symlinked directory.
fn collect_videos(root: &Path, dir: &Path, out: &mut Vec<PathBuf>, visited: &mut HashSet<PathBuf>) {
    // Record this directory's canonical path; if we've already visited it,
    // a symlink cycle has brought us back — stop.
    if let Ok(canon) = std::fs::canonicalize(dir) {
        if !visited.insert(canon) {
            return;
        }
    }

    let mut entries: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd.filter_map(|e| e.ok()).map(|e| e.path()).collect(),
        Err(_) => return,
    };
    entries.sort();

    for entry in entries {
        // Skip hidden directories/files at any level
        let component_name = entry.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if component_name.starts_with('.') {
            continue;
        }

        // Determine whether this entry is itself a symlink. We FOLLOW symlinks
        // that resolve to regular files, but never descend into a symlinked
        // DIRECTORY: that can point back up the tree
        // (cycle) or escape the root entirely. (`is_dir`/`is_file` below follow
        // symlinks, so they report on the link's target.)
        let is_symlink = std::fs::symlink_metadata(&entry)
            .map(|m| m.file_type().is_symlink())
            .unwrap_or(false);

        if entry.is_dir() {
            // Skip symlinked directories for cycle safety; the `visited` set
            // guards real-path recursion below.
            if is_symlink {
                continue;
            }
            collect_videos(root, &entry, out, visited);
        } else if entry.is_file() && has_video_extension(&entry) && !is_in_place_temp(&entry) {
            // Verify no hidden component in relative path
            if let Ok(rel) = entry.strip_prefix(root) {
                let has_hidden = rel.components().any(|c| {
                    c.as_os_str()
                        .to_str()
                        .map(|s| s.starts_with('.'))
                        .unwrap_or(false)
                });
                if !has_hidden {
                    out.push(entry);
                }
            }
        }
    }
}

/// Create a temporary working directory preserving the relative path structure.
///
/// For a video at `~/Anime/Show/S1/ep01.mkv` with root `~/Anime`:
/// returns `~/Anime/.translate_temp/Show/S1/ep01/`.
///
/// Always creates `candidates/` and `reference/` subdirectories inside.
pub fn create_work_dir(video_path: &Path, root_input: &Path) -> Result<PathBuf> {
    let relative = video_path
        .parent()
        .and_then(|p| p.strip_prefix(root_input).ok())
        .unwrap_or(Path::new(""));

    let stem = video_path.file_stem().unwrap_or_default();

    let temp_root = root_input.join(".translate_temp");
    let work_dir = temp_root.join(relative).join(stem);

    std::fs::create_dir_all(work_dir.join("candidates"))?;
    std::fs::create_dir_all(work_dir.join("reference"))?;

    Ok(work_dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn touch(path: &Path) {
        std::fs::File::create(path).unwrap();
    }

    // --- find_videos ---

    #[test]
    fn nonexistent_path_returns_empty() {
        let result = find_videos(Path::new("/nonexistent/path/that/does/not/exist"));
        assert!(result.is_empty());
    }

    #[test]
    fn single_mkv_file_returns_itself() {
        let dir = TempDir::new().unwrap();
        let video = dir.path().join("episode.mkv");
        touch(&video);
        assert_eq!(find_videos(&video), vec![video]);
    }

    #[test]
    fn single_mp4_file_returns_itself() {
        let dir = TempDir::new().unwrap();
        let video = dir.path().join("movie.mp4");
        touch(&video);
        assert_eq!(find_videos(&video), vec![video]);
    }

    #[test]
    fn non_video_file_returns_empty() {
        let dir = TempDir::new().unwrap();
        let srt = dir.path().join("subtitles.srt");
        touch(&srt);
        assert!(find_videos(&srt).is_empty());
    }

    #[test]
    fn in_place_temp_is_skipped() {
        let dir = TempDir::new().unwrap();
        let temp = dir.path().join("Episode01.translating.mkv");
        touch(&temp);
        assert!(find_videos(&temp).is_empty());
    }

    #[test]
    fn directory_finds_all_videos_sorted() {
        let dir = TempDir::new().unwrap();
        let b = dir.path().join("b.mkv");
        let a = dir.path().join("a.mkv");
        let c = dir.path().join("c.mp4");
        touch(&a);
        touch(&b);
        touch(&c);
        let result = find_videos(dir.path());
        assert_eq!(result, vec![a, b, c]);
    }

    #[test]
    fn hidden_directories_are_skipped() {
        let dir = TempDir::new().unwrap();
        let hidden = dir.path().join(".hidden");
        std::fs::create_dir_all(&hidden).unwrap();
        touch(&hidden.join("video.mkv"));
        let normal = dir.path().join("video.mkv");
        touch(&normal);
        let result = find_videos(dir.path());
        assert_eq!(result, vec![normal]);
    }

    #[test]
    fn translating_temp_skipped_in_directory() {
        let dir = TempDir::new().unwrap();
        let temp = dir.path().join("Ep01.translating.mkv");
        let real = dir.path().join("Ep01.mkv");
        touch(&temp);
        touch(&real);
        let result = find_videos(dir.path());
        assert_eq!(result, vec![real]);
    }

    #[test]
    fn recursive_subdirectories() {
        let dir = TempDir::new().unwrap();
        let sub = dir.path().join("Season1");
        std::fs::create_dir_all(&sub).unwrap();
        let v1 = sub.join("ep01.mkv");
        let v2 = dir.path().join("movie.mp4");
        touch(&v1);
        touch(&v2);
        let result = find_videos(dir.path());
        assert_eq!(result.len(), 2);
        assert!(result.contains(&v1));
        assert!(result.contains(&v2));
    }

    #[cfg(unix)]
    #[test]
    fn self_referential_symlink_does_not_loop() {
        use std::os::unix::fs::symlink;

        let dir = TempDir::new().unwrap();
        let real = dir.path().join("ep01.mkv");
        touch(&real);

        // Create a directory containing a symlink that points back to its parent,
        // forming a cycle. Without cycle detection this recurses forever.
        let sub = dir.path().join("sub");
        std::fs::create_dir_all(&sub).unwrap();
        symlink(dir.path(), sub.join("loop")).unwrap();

        let result = find_videos(dir.path());
        // Must terminate and find the single real video exactly once.
        assert_eq!(result, vec![real]);
    }

    #[cfg(unix)]
    #[test]
    fn symlink_to_real_file_is_included() {
        use std::os::unix::fs::symlink;

        // The real .mkv lives outside the scanned tree; only a symlink to it
        // sits inside. File symlinks are followed, so it must be discovered.
        let target_dir = TempDir::new().unwrap();
        let real = target_dir.path().join("real.mkv");
        touch(&real);

        let scan_dir = TempDir::new().unwrap();
        let link = scan_dir.path().join("linked.mkv");
        symlink(&real, &link).unwrap();

        let result = find_videos(scan_dir.path());
        assert_eq!(result, vec![link]);
    }

    // --- create_work_dir ---

    #[test]
    fn creates_expected_structure() {
        let dir = TempDir::new().unwrap();
        let video = dir.path().join("Show").join("ep01.mkv");
        std::fs::create_dir_all(video.parent().unwrap()).unwrap();
        touch(&video);

        let work_dir = create_work_dir(&video, dir.path()).unwrap();
        let expected = dir.path().join(".translate_temp").join("Show").join("ep01");
        assert_eq!(work_dir, expected);
        assert!(work_dir.join("candidates").exists());
        assert!(work_dir.join("reference").exists());
    }

    #[test]
    fn video_directly_in_root() {
        let dir = TempDir::new().unwrap();
        let video = dir.path().join("movie.mkv");
        touch(&video);

        let work_dir = create_work_dir(&video, dir.path()).unwrap();
        let expected = dir.path().join(".translate_temp").join("movie");
        assert_eq!(work_dir, expected);
        assert!(work_dir.join("candidates").exists());
        assert!(work_dir.join("reference").exists());
    }
}
