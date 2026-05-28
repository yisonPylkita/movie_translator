//! Embedded-CPython backend for the ML stages.
//!
//! Instead of spawning `python ml/*.py` subprocesses, this module embeds
//! CPython via PyO3 inside the Rust binary and calls into the
//! `movie_translator` Python package directly. Model objects (e.g.
//! `SubtitleTranslator`) are loaded once per binary run and reused across
//! every file in a `run_all`.
//!
//! # Thread safety
//! All entry points acquire the GIL via [`Python::attach`]. ML stages are
//! already serialised onto a single GPU worker by `mt-pipeline`, so GIL
//! contention is a non-issue for the heavy paths. The lightweight filename
//! parser (called from parallel discovery) is GIL-bound — that's acceptable.
//!
//! # `sys.path` bootstrap
//! On first GIL acquire we prepend `MT_REPO_ROOT` (if set) and the
//! conventional repo root (resolved by walking up from `current_exe` /
//! `current_dir` looking for a `movie_translator/` package) so
//! `import movie_translator` resolves regardless of the cwd the binary was
//! invoked from.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use mt_core::{BoundingBox, BurnedInResult, DialogueLine, MtError, OCRResult, Result};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyModule, PyString, PyTuple};

/// Lazily-initialised holder for cached Python module handles.
///
/// PyO3 0.28 disallows holding `Py<…>` references across GIL drops via plain
/// statics without `Send`-only types; `Py<PyAny>` is `Send + Sync`, so we keep
/// them inside a `Mutex<Option<Modules>>` initialised on first call.
struct Modules {
    /// `movie_translator.types` — for constructing `DialogueLine` / `OCRResult`.
    types: Py<PyAny>,
    /// `movie_translator.translation` — translator + `translate_dialogue_lines`.
    translation: Py<PyAny>,
    /// `movie_translator.ocr.pgs_extractor` — `extract_pgs_track`.
    pgs_extractor: Py<PyAny>,
    /// `movie_translator.ocr` — `extract_burned_in_subtitles`.
    ocr: Py<PyAny>,
    /// `movie_translator.inpainting` — `remove_burned_in_subtitles`.
    inpainting: Py<PyAny>,
    /// `movie_translator.identifier.parser` — `parse_filename`.
    parser: Py<PyAny>,
}

/// Shared, lazily-loaded module handles, accessed via [`with_modules`].
fn modules_cell() -> &'static Mutex<Option<Modules>> {
    static MODULES: OnceLock<Mutex<Option<Modules>>> = OnceLock::new();
    MODULES.get_or_init(|| Mutex::new(None))
}

/// Run `f` with cloned handles to every cached Python module. Locks the
/// modules mutex only for the duration of the clone, so callers can freely
/// re-enter the backend (e.g. `translate` calling `model_cache` which calls
/// back into `modules_cell`) without deadlocking.
fn with_modules<F, R>(py: Python<'_>, f: F) -> Result<R>
where
    F: FnOnce(Python<'_>, Modules) -> Result<R>,
{
    let cell = modules_cell();
    // Fast path: take cloned handles under the lock and release immediately.
    {
        let guard = cell.lock().expect("modules mutex poisoned");
        if let Some(m) = guard.as_ref() {
            let cloned = m.clone_handles(py);
            drop(guard);
            return f(py, cloned);
        }
    }

    // Slow path: first-time init. Must not hold the lock while importing,
    // because the Python imports can in principle re-enter the backend.
    bootstrap_sys_path(py)?;

    let types = py_import(py, "movie_translator.types")?;
    let translation = py_import(py, "movie_translator.translation")?;
    let pgs_extractor = py_import(py, "movie_translator.ocr.pgs_extractor")?;
    let ocr = py_import(py, "movie_translator.ocr")?;
    let inpainting = py_import(py, "movie_translator.inpainting")?;
    let parser = py_import(py, "movie_translator.identifier.parser")?;
    let m = Modules {
        types,
        translation,
        pgs_extractor,
        ocr,
        inpainting,
        parser,
    };

    let mut guard = cell.lock().expect("modules mutex poisoned");
    // Race: another thread may have initialised while we imported. Either
    // way the result is equivalent — keep whichever is there.
    if guard.is_none() {
        *guard = Some(m.clone_handles(py));
    }
    let cloned = guard.as_ref().unwrap().clone_handles(py);
    drop(guard);
    f(py, cloned)
}

impl Modules {
    fn clone_handles(&self, py: Python<'_>) -> Self {
        Modules {
            types: self.types.clone_ref(py),
            translation: self.translation.clone_ref(py),
            pgs_extractor: self.pgs_extractor.clone_ref(py),
            ocr: self.ocr.clone_ref(py),
            inpainting: self.inpainting.clone_ref(py),
            parser: self.parser.clone_ref(py),
        }
    }
}

/// Cached `ModelCache` Python instance, kept alive for the whole binary run
/// so models load only once across every file processed by `run_all`.
fn model_cache(py: Python<'_>, translation: &Py<PyAny>) -> Result<Py<PyAny>> {
    static CACHE: OnceLock<Mutex<Option<Py<PyAny>>>> = OnceLock::new();
    let cell = CACHE.get_or_init(|| Mutex::new(None));
    {
        let guard = cell.lock().expect("model cache mutex poisoned");
        if let Some(c) = guard.as_ref() {
            return Ok(c.clone_ref(py));
        }
    }
    let model_cache_cls = translation.bind(py).getattr("ModelCache").map_err(py_err)?;
    let instance = model_cache_cls.call0().map_err(py_err)?;
    let py_obj: Py<PyAny> = instance.into();
    let mut guard = cell.lock().expect("model cache mutex poisoned");
    if guard.is_none() {
        *guard = Some(py_obj.clone_ref(py));
    }
    Ok(guard.as_ref().unwrap().clone_ref(py))
}

/// Resolve a likely repo root by walking up from a starting path looking for
/// a `movie_translator/__init__.py`.
fn locate_package_root_from(start: &Path) -> Option<PathBuf> {
    let mut dir = Some(start);
    while let Some(d) = dir {
        if d.join("movie_translator").join("__init__.py").is_file() {
            return Some(d.to_path_buf());
        }
        dir = d.parent();
    }
    None
}

/// Locate the repo root containing the `movie_translator/` Python package
/// so we can put it on `sys.path` and `import movie_translator`.
fn locate_repo_root() -> Option<PathBuf> {
    if let Ok(v) = std::env::var("MT_REPO_ROOT") {
        let p = PathBuf::from(v);
        if p.join("movie_translator").join("__init__.py").is_file() {
            return Some(p);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        let start = exe.parent().unwrap_or(&exe);
        if let Some(r) = locate_package_root_from(start) {
            return Some(r);
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        if let Some(r) = locate_package_root_from(&cwd) {
            return Some(r);
        }
    }
    None
}

/// Locate the venv site-packages directory next to the repo root.
///
/// PyO3 embeds the base Python interpreter (compiled into the binary at
/// `cargo build` time via `PYO3_PYTHON`), which does NOT see venv-installed
/// packages by default — the venv's site-packages must be added explicitly.
///
/// Resolution order:
///  1. `MT_VENV_SITE_PACKAGES` env var (explicit override).
///  2. `<repo_root>/.venv/lib/python<X.Y>/site-packages` if it exists,
///     where `<X.Y>` is the embedded interpreter's `sys.version_info`.
fn locate_venv_site_packages(py: Python<'_>, repo_root: &Path) -> Option<PathBuf> {
    if let Ok(v) = std::env::var("MT_VENV_SITE_PACKAGES") {
        let p = PathBuf::from(v);
        if p.is_dir() {
            return Some(p);
        }
    }
    // Ask the interpreter for its own major.minor (e.g. "3.14") so the path is
    // correct even on a Python upgrade.
    let sys = py.import("sys").ok()?;
    let version_info = sys.getattr("version_info").ok()?;
    let major: u32 = version_info.getattr("major").ok()?.extract().ok()?;
    let minor: u32 = version_info.getattr("minor").ok()?.extract().ok()?;
    let site_packages = repo_root
        .join(".venv")
        .join("lib")
        .join(format!("python{major}.{minor}"))
        .join("site-packages");
    if site_packages.is_dir() {
        Some(site_packages)
    } else {
        None
    }
}

/// Prepend the repo root (for `import movie_translator`) and the venv's
/// site-packages (for torch/transformers/guessit/…) to `sys.path`.
///
/// Without the venv site-packages, the embedded interpreter only sees the
/// base Python's stdlib — the moment we `import movie_translator.logging`
/// (which pulls `rich`) the import explodes. This bootstrap is what makes
/// PyO3 embedding usable with a uv-managed venv.
fn bootstrap_sys_path(py: Python<'_>) -> Result<()> {
    let repo_root = locate_repo_root().ok_or_else(|| {
        MtError::PathResolution(
            "could not locate movie_translator/ package; set MT_REPO_ROOT to the repo root"
                .to_string(),
        )
    })?;
    let sys = py.import("sys").map_err(py_err)?;
    let path: Bound<'_, PyList> = sys
        .getattr("path")
        .map_err(py_err)?
        .cast_into()
        .map_err(|e| MtError::Parse(format!("sys.path is not a list: {e}")))?;

    // Insert venv site-packages first (so it shadows the base interpreter's
    // bare stdlib), then the repo root (so `import movie_translator` resolves).
    let mut to_prepend: Vec<PathBuf> = Vec::new();
    if let Some(sp) = locate_venv_site_packages(py, &repo_root) {
        to_prepend.push(sp);
    }
    to_prepend.push(repo_root);

    // Avoid duplicates: only insert if not already present.
    let existing: Vec<String> = path
        .iter()
        .filter_map(|item| item.extract::<String>().ok())
        .collect();
    // Prepend in REVERSE so the final order matches `to_prepend` (insert(0)).
    for entry in to_prepend.iter().rev() {
        let entry_s = entry.to_string_lossy().to_string();
        if !existing.iter().any(|e| e == &entry_s) {
            let s = PyString::new(py, &entry_s);
            path.insert(0, s).map_err(py_err)?;
        }
    }
    Ok(())
}

/// Import a Python module and return its handle.
fn py_import(py: Python<'_>, name: &str) -> Result<Py<PyAny>> {
    let module: Bound<'_, PyModule> = py.import(name).map_err(py_err)?;
    Ok(module.into_any().unbind())
}

/// Translate a `PyErr` into an `MtError::Parse` carrying the traceback text.
fn py_err(err: PyErr) -> MtError {
    Python::attach(|py| {
        let traceback = err
            .traceback(py)
            .and_then(|tb| tb.format().ok())
            .unwrap_or_default();
        MtError::Parse(format!("python error: {err}\n{traceback}"))
    })
}

/// Build a Python `DialogueLine` NamedTuple from a Rust [`DialogueLine`].
fn dialogue_to_py<'py>(
    types_mod: &Bound<'py, PyAny>,
    line: &DialogueLine,
) -> Result<Bound<'py, PyAny>> {
    let cls = types_mod.getattr("DialogueLine").map_err(py_err)?;
    let args = (line.start_ms, line.end_ms, &line.text);
    cls.call1(args).map_err(py_err)
}

/// Convert a Python `DialogueLine` NamedTuple back to a Rust [`DialogueLine`].
fn dialogue_from_py(obj: &Bound<'_, PyAny>) -> Result<DialogueLine> {
    let start_ms: i64 = obj
        .getattr("start_ms")
        .map_err(py_err)?
        .extract()
        .map_err(py_err)?;
    let end_ms: i64 = obj
        .getattr("end_ms")
        .map_err(py_err)?
        .extract()
        .map_err(py_err)?;
    let text: String = obj
        .getattr("text")
        .map_err(py_err)?
        .extract()
        .map_err(py_err)?;
    Ok(DialogueLine {
        start_ms,
        end_ms,
        text,
    })
}

/// Convert a Python `BoundingBox` NamedTuple to Rust.
fn bbox_from_py(obj: &Bound<'_, PyAny>) -> Result<BoundingBox> {
    let x: f64 = obj
        .getattr("x")
        .map_err(py_err)?
        .extract()
        .map_err(py_err)?;
    let y: f64 = obj
        .getattr("y")
        .map_err(py_err)?
        .extract()
        .map_err(py_err)?;
    let width: f64 = obj
        .getattr("width")
        .map_err(py_err)?
        .extract()
        .map_err(py_err)?;
    let height: f64 = obj
        .getattr("height")
        .map_err(py_err)?
        .extract()
        .map_err(py_err)?;
    Ok(BoundingBox {
        x,
        y,
        width,
        height,
    })
}

/// Convert a Python `OCRResult` NamedTuple to Rust.
fn ocr_from_py(obj: &Bound<'_, PyAny>) -> Result<OCRResult> {
    let timestamp_ms: i64 = obj
        .getattr("timestamp_ms")
        .map_err(py_err)?
        .extract()
        .map_err(py_err)?;
    let text: String = obj
        .getattr("text")
        .map_err(py_err)?
        .extract()
        .map_err(py_err)?;
    let boxes_obj = obj.getattr("boxes").map_err(py_err)?;
    let mut boxes = Vec::new();
    for b in boxes_obj.try_iter().map_err(py_err)? {
        let b = b.map_err(py_err)?;
        boxes.push(bbox_from_py(&b)?);
    }
    Ok(OCRResult {
        timestamp_ms,
        text,
        boxes,
    })
}

/// Build a Python `BoundingBox` NamedTuple from Rust.
fn bbox_to_py<'py>(types_mod: &Bound<'py, PyAny>, b: &BoundingBox) -> Result<Bound<'py, PyAny>> {
    let cls = types_mod.getattr("BoundingBox").map_err(py_err)?;
    cls.call1((b.x, b.y, b.width, b.height)).map_err(py_err)
}

/// Build a Python `OCRResult` NamedTuple from Rust.
fn ocr_to_py<'py>(
    py: Python<'py>,
    types_mod: &Bound<'py, PyAny>,
    r: &OCRResult,
) -> Result<Bound<'py, PyAny>> {
    let cls = types_mod.getattr("OCRResult").map_err(py_err)?;
    let boxes = PyList::empty(py);
    for b in &r.boxes {
        boxes.append(bbox_to_py(types_mod, b)?).map_err(py_err)?;
    }
    cls.call1((r.timestamp_ms, &r.text, boxes)).map_err(py_err)
}

/// Translate dialogue lines through the embedded Python translator. The
/// underlying `ModelCache` keeps the model loaded across calls.
pub fn translate(
    lines: &[DialogueLine],
    device: &str,
    batch_size: u32,
    model: &str,
    proper_nouns: Option<&[String]>,
) -> Result<Vec<DialogueLine>> {
    Python::attach(|py| {
        with_modules(py, |py, m| {
            let types_bound = m.types.bind(py);
            let translation_bound = m.translation.bind(py);

            let py_lines = PyList::empty(py);
            for line in lines {
                py_lines
                    .append(dialogue_to_py(types_bound, line)?)
                    .map_err(py_err)?;
            }

            let proper = match proper_nouns {
                Some(nouns) if !nouns.is_empty() => {
                    let set_cls = py
                        .import("builtins")
                        .map_err(py_err)?
                        .getattr("set")
                        .map_err(py_err)?;
                    let lst = PyList::empty(py);
                    for n in nouns {
                        lst.append(n).map_err(py_err)?;
                    }
                    set_cls.call1((lst,)).map_err(py_err)?.into_any()
                }
                _ => py.None().bind(py).clone(),
            };

            let func = translation_bound
                .getattr("translate_dialogue_lines")
                .map_err(py_err)?;

            let cache = model_cache(py, &m.translation)?;
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs
                .set_item("dialogue_lines", &py_lines)
                .map_err(py_err)?;
            kwargs.set_item("device", device).map_err(py_err)?;
            kwargs.set_item("batch_size", batch_size).map_err(py_err)?;
            kwargs.set_item("model", model).map_err(py_err)?;
            kwargs.set_item("proper_nouns", &proper).map_err(py_err)?;
            kwargs
                .set_item("model_cache", cache.bind(py))
                .map_err(py_err)?;

            let result = func
                .call(PyTuple::empty(py), Some(&kwargs))
                .map_err(py_err)?;

            let mut out = Vec::with_capacity(lines.len());
            for item in result.try_iter().map_err(py_err)? {
                let item = item.map_err(py_err)?;
                out.push(dialogue_from_py(&item)?);
            }
            Ok(out)
        })
    })
}

/// Extract a PGS subtitle track to SRT via embedded Python.
pub fn ocr_pgs(video: &Path, track_index: u32, work_dir: &Path) -> Result<Option<PathBuf>> {
    Python::attach(|py| {
        with_modules(py, |py, m| {
            let pgs_bound = m.pgs_extractor.bind(py);

            let pathlib = py.import("pathlib").map_err(py_err)?;
            let path_cls = pathlib.getattr("Path").map_err(py_err)?;
            let video_p = path_cls
                .call1((video.to_string_lossy().as_ref(),))
                .map_err(py_err)?;
            let work_p = path_cls
                .call1((work_dir.to_string_lossy().as_ref(),))
                .map_err(py_err)?;

            let func = pgs_bound.getattr("extract_pgs_track").map_err(py_err)?;
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("video_path", &video_p).map_err(py_err)?;
            kwargs
                .set_item("track_index", track_index)
                .map_err(py_err)?;
            kwargs.set_item("work_dir", &work_p).map_err(py_err)?;

            let result = func
                .call(PyTuple::empty(py), Some(&kwargs))
                .map_err(py_err)?;
            if result.is_none() {
                return Ok(None);
            }
            let s: String = result.str().map_err(py_err)?.extract().map_err(py_err)?;
            Ok(Some(PathBuf::from(s)))
        })
    })
}

/// Extract burned-in subtitles via OCR through embedded Python.
pub fn ocr_burned_in(
    video: &Path,
    output_dir: &Path,
    crop_ratio: f64,
    fps: u32,
) -> Result<BurnedInResult> {
    Python::attach(|py| {
        with_modules(py, |py, m| {
            let ocr_bound = m.ocr.bind(py);

            let pathlib = py.import("pathlib").map_err(py_err)?;
            let path_cls = pathlib.getattr("Path").map_err(py_err)?;
            let video_p = path_cls
                .call1((video.to_string_lossy().as_ref(),))
                .map_err(py_err)?;
            let out_p = path_cls
                .call1((output_dir.to_string_lossy().as_ref(),))
                .map_err(py_err)?;

            let func = ocr_bound
                .getattr("extract_burned_in_subtitles")
                .map_err(py_err)?;
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("video_path", &video_p).map_err(py_err)?;
            kwargs.set_item("output_dir", &out_p).map_err(py_err)?;
            kwargs.set_item("crop_ratio", crop_ratio).map_err(py_err)?;
            kwargs.set_item("fps", fps).map_err(py_err)?;

            let result = func
                .call(PyTuple::empty(py), Some(&kwargs))
                .map_err(py_err)?;
            if result.is_none() {
                return Err(MtError::Parse(
                    "extract_burned_in_subtitles returned None".into(),
                ));
            }

            let srt_path: String = result
                .getattr("srt_path")
                .map_err(py_err)?
                .str()
                .map_err(py_err)?
                .extract()
                .map_err(py_err)?;
            let mut results = Vec::new();
            for item in result
                .getattr("ocr_results")
                .map_err(py_err)?
                .try_iter()
                .map_err(py_err)?
            {
                let item = item.map_err(py_err)?;
                results.push(ocr_from_py(&item)?);
            }
            Ok(BurnedInResult {
                srt_path: PathBuf::from(srt_path),
                ocr_results: results,
            })
        })
    })
}

/// Remove burned-in subtitles via inpainting through embedded Python.
pub fn inpaint(
    video: &Path,
    output: &Path,
    device: &str,
    backend: &str,
    ocr_results: &[OCRResult],
) -> Result<PathBuf> {
    Python::attach(|py| {
        with_modules(py, |py, m| {
            let types_bound = m.types.bind(py);
            let inpainting_bound = m.inpainting.bind(py);

            let pathlib = py.import("pathlib").map_err(py_err)?;
            let path_cls = pathlib.getattr("Path").map_err(py_err)?;
            let video_p = path_cls
                .call1((video.to_string_lossy().as_ref(),))
                .map_err(py_err)?;
            let out_p = path_cls
                .call1((output.to_string_lossy().as_ref(),))
                .map_err(py_err)?;

            let py_results = PyList::empty(py);
            for r in ocr_results {
                py_results
                    .append(ocr_to_py(py, types_bound, r)?)
                    .map_err(py_err)?;
            }

            let func = inpainting_bound
                .getattr("remove_burned_in_subtitles")
                .map_err(py_err)?;
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("video_path", &video_p).map_err(py_err)?;
            kwargs.set_item("output_path", &out_p).map_err(py_err)?;
            kwargs
                .set_item("ocr_results", &py_results)
                .map_err(py_err)?;
            kwargs.set_item("device", device).map_err(py_err)?;
            kwargs.set_item("backend", backend).map_err(py_err)?;

            func.call(PyTuple::empty(py), Some(&kwargs))
                .map_err(py_err)?;
            Ok(output.to_path_buf())
        })
    })
}

/// Parsed filename metadata returned by [`parse_filename`].
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedFilename {
    pub title: Option<String>,
    pub year: Option<i32>,
    pub season: Option<i32>,
    pub episode: Option<i32>,
    pub media_type: String,
    pub is_anime: bool,
    pub release_group: Option<String>,
}

/// Parse a video filename through `movie_translator.identifier.parser`.
pub fn parse_filename(filename: &str, folder_name: Option<&str>) -> Result<ParsedFilename> {
    Python::attach(|py| {
        with_modules(py, |py, m| {
            let parser_bound = m.parser.bind(py);

            let func = parser_bound.getattr("parse_filename").map_err(py_err)?;
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("filename", filename).map_err(py_err)?;
            match folder_name {
                Some(f) => kwargs.set_item("folder_name", f).map_err(py_err)?,
                None => kwargs.set_item("folder_name", py.None()).map_err(py_err)?,
            }
            let dict = func
                .call(PyTuple::empty(py), Some(&kwargs))
                .map_err(py_err)?;

            let title: Option<String> = extract_opt(&dict, "title")?;
            let year: Option<i32> = extract_opt_int(&dict, "year")?;
            let season: Option<i32> = extract_opt_int(&dict, "season")?;
            let episode: Option<i32> = extract_opt_int(&dict, "episode")?;
            let media_type: String = dict
                .get_item("media_type")
                .map_err(py_err)?
                .extract()
                .map_err(py_err)?;
            let is_anime: bool = dict
                .get_item("is_anime")
                .map_err(py_err)?
                .extract()
                .map_err(py_err)?;
            let release_group: Option<String> = extract_opt(&dict, "release_group")?;

            Ok(ParsedFilename {
                title,
                year,
                season,
                episode,
                media_type,
                is_anime,
                release_group,
            })
        })
    })
}

/// Check whether the Apple Vision OCR bindings (Vision + Quartz) import
/// cleanly in the embedded interpreter. Cached on first success/failure for
/// the lifetime of the process.
pub fn vision_ocr_available() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        Python::attach(|py| py.import("Vision").is_ok() && py.import("Quartz").is_ok())
    })
}

/// Extract a string from a Python dict, returning `None` if absent or Py-None.
fn extract_opt(dict: &Bound<'_, PyAny>, key: &str) -> Result<Option<String>> {
    let v = dict.get_item(key).map_err(py_err)?;
    if v.is_none() {
        Ok(None)
    } else {
        Ok(Some(v.extract().map_err(py_err)?))
    }
}

/// Extract an int from a Python dict, returning `None` if absent or Py-None.
fn extract_opt_int(dict: &Bound<'_, PyAny>, key: &str) -> Result<Option<i32>> {
    let v = dict.get_item(key).map_err(py_err)?;
    if v.is_none() {
        Ok(None)
    } else {
        // guessit can yield various int-like types; force through int().
        let int_obj = py_int(v.py(), &v)?;
        Ok(Some(int_obj.extract().map_err(py_err)?))
    }
}

/// Coerce any int-like Python object through `int(x)`.
fn py_int<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> Result<Bound<'py, PyAny>> {
    let int_cls = py
        .import("builtins")
        .map_err(py_err)?
        .getattr("int")
        .map_err(py_err)?;
    int_cls.call1((obj,)).map_err(py_err)
}
