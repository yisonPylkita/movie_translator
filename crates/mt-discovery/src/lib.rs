//! Media file discovery and filesystem scanning.

pub mod discovery;
pub mod hasher;
pub mod identify;
pub mod metadata;
pub mod parser;
pub mod tmdb;

// Public API re-exports
pub use discovery::{create_work_dir, find_videos};
pub use hasher::{compute_napiprojekt_hash, compute_oshash};
pub use identify::identify_media;
pub use parser::{parse_filename, ParsedName};
