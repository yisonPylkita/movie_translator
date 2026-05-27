# Movie Translator development tasks
#
# Orchestration is Rust (cargo workspace, crates/mt-*); machine-learning
# inference stays in Python under ml/ (invoked by the Rust binary).

default:
    @just --list

# --- Rust (primary: CLI + orchestration) ---

# Build the release binary (target/release/movie-translator)
build:
    cargo build --release --bin movie-translator

# Run the Rust test suite
test *args:
    cargo test --workspace {{ args }}

# Lint + format check, no modifications (mirrors CI)
check:
    cargo clippy --workspace --all-targets -- -D warnings
    cargo fmt --check
    uv run ruff check ml/

# Auto-fix lint + format (Rust + Python ML scripts)
lint:
    cargo clippy --workspace --all-targets --fix --allow-dirty --allow-staged -- -D warnings
    cargo fmt
    uv run ruff check --fix ml/
    uv run ruff format ml/

# Run all checks and tests (CI equivalent)
ci: check test

# Run the movie-translator binary. Usage: just run <file-or-dir> [args...]
run dir *args:
    cargo run --release --bin movie-translator -- "{{ dir }}" {{ args }}

# --- Python ML backend (translation / OCR / inpainting) ---

# Install / sync the Python environment used by the ml/ scripts
sync:
    uv sync

# Run the Python ML-backend test suite (translation/ocr/inpainting modules)
py-test *args:
    uv run pytest -v movie_translator/translation movie_translator/ocr movie_translator/inpainting {{ args }}

# Install git pre-commit hook (runs `just check`)
install-hooks:
    @echo '#!/bin/sh' > .git/hooks/pre-commit
    @echo 'just check' >> .git/hooks/pre-commit
    @chmod +x .git/hooks/pre-commit
    @echo 'Pre-commit hook installed (runs just check).'
