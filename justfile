# Movie Translator — English→Polish video subtitle translator.
#
# `just` is the entry point for everything: setup, build, run, tests, lint.
# Run `just` (or `just --list`) to see every available recipe.

default:
    @just --list

# ─── Setup ─────────────────────────────────────────────────────────────────

# Full one-shot setup: Python env, submodules, model files, and the binary.
# Idempotent — safe to re-run. On macOS, run `just brew` once first.
setup: deps submodules model build
    @echo
    @echo "Setup complete. Try: just run /path/to/video.mkv --dry-run"

# macOS only: install system tools (just, ffmpeg, git-lfs, uv, ...) via Homebrew.
brew:
    brew bundle

# Install / sync the Python ML backend environment (torch, transformers, ...).
deps:
    uv sync

# Initialise git submodules (vendor/ilass).
submodules:
    git submodule update --init --recursive

# Pull the translation model files via git-lfs (Allegro BiDi en↔pl).
model:
    git lfs install
    git lfs pull

# ─── Build ─────────────────────────────────────────────────────────────────

# Build the release binary + the vendored ilass alignment engine.
build:
    cargo build --release --bin movie-translator
    cd vendor/ilass && cargo build --release

# Remove all Rust build artifacts.
clean:
    cargo clean

# ─── Run ───────────────────────────────────────────────────────────────────

# Translate videos: `just run <file-or-dir> [flags]` (run with --help for all flags).
run input *args:
    cargo run --release --quiet --bin movie-translator -- "{{ input }}" {{ args }}

# Extract subtitles (text + burned-in OCR), no translation: `just extract <file-or-dir>`.
extract input *args:
    cargo run --release --quiet --bin movie-translator -- extract "{{ input }}" {{ args }}

# ─── Tests + lint ──────────────────────────────────────────────────────────

# Run the Rust test suite. Usage: `just test [extra cargo-test args]`
test *args:
    cargo test --workspace {{ args }}

# Run the Python ML-backend test suite.
py-test *args:
    uv run pytest -o addopts="" movie_translator {{ args }}

# Lint + format check, no modifications (mirrors CI).
check:
    cargo clippy --workspace --all-targets -- -D warnings
    cargo fmt --check
    uv run ruff check ml/ movie_translator/

# Auto-fix lint + format issues (Rust + Python).
lint:
    cargo clippy --workspace --all-targets --fix --allow-dirty --allow-staged -- -D warnings
    cargo fmt
    uv run ruff check --fix ml/ movie_translator/
    uv run ruff format ml/ movie_translator/

# All checks + all tests (CI equivalent).
ci: check test

# ─── Misc ──────────────────────────────────────────────────────────────────

# Install a git pre-commit hook that runs `just check`.
install-hooks:
    @echo '#!/bin/sh' > .git/hooks/pre-commit
    @echo 'just check' >> .git/hooks/pre-commit
    @chmod +x .git/hooks/pre-commit
    @echo 'Pre-commit hook installed (runs just check).'
