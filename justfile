# Movie Translator — English→Polish video subtitle translator.
#
# `just` is the entry point for everything: setup, build, run, tests, lint.
# Run `just` (or `just --list`) to see every available recipe.

# PyO3 needs to know which Python interpreter to embed at build time. We
# point it at the uv-managed venv so the embedded interpreter sees the same
# packages the Python code uses (torch, transformers, guessit, …).
# `just deps` creates the venv before any build runs.
export PYO3_PYTHON := justfile_directory() + "/.venv/bin/python"

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

# Download + convert the translation model (Allegro BiDi en↔pl) to MLX INT8.
#
# Downloads the PyTorch model from HuggingFace (allegro/BiDi-eng-pol), converts
# it to MLX, quantises to INT8, and saves to models/allegro/.
#
# On first run, use `--torch-dir` if you already have the model cached, or
# omit it to download from HuggingFace automatically.
model:
    uv run python scripts/download_and_convert_model.py

# Pull the pre-converted MLX INT8 model from Git LFS (faster than re-converting).
model-pull:
    git lfs install
    git lfs pull --include="models/allegro/"

# ─── Build ─────────────────────────────────────────────────────────────────

# Build the release binary + the vendored ilass alignment engine.
# Depends on `deps` so the venv exists before cargo links against libpython.
build: deps
    cargo build --release --bin movie-translator --bin anime-dl
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

# Download a whole anime season from ogladajanime.pl: `just anime-dl "<name>" [flags]`.
anime-dl name *args:
    cargo run --release --quiet --bin anime-dl -- "{{ name }}" {{ args }}

# ─── Tests + lint ──────────────────────────────────────────────────────────

# Run the Rust test suite. Usage: `just test [extra cargo-test args]`
test *args: deps
    cargo test --workspace {{ args }}

# Run the Python ML-backend test suite.
py-test *args:
    uv run pytest -o addopts="" movie_translator {{ args }}

# Lint + format check, no modifications (mirrors CI).
check: deps
    cargo clippy --workspace --all-targets -- -D warnings
    cargo fmt --check
    uv run ruff check movie_translator/

# Auto-fix lint + format issues (Rust + Python).
lint: deps
    cargo clippy --workspace --all-targets --fix --allow-dirty --allow-staged -- -D warnings
    cargo fmt
    uv run ruff check --fix movie_translator/
    uv run ruff format movie_translator/

# All checks + all tests (CI equivalent).
ci: check test

# ─── Misc ──────────────────────────────────────────────────────────────────

# Install a git pre-commit hook that runs `just check`.
install-hooks:
    @echo '#!/bin/sh' > .git/hooks/pre-commit
    @echo 'just check' >> .git/hooks/pre-commit
    @chmod +x .git/hooks/pre-commit
    @echo 'Pre-commit hook installed (runs just check).'
