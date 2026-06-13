# Movie Translator — English→Polish video subtitle translator.
#
# `just` is the entry point for everything: setup, build, run, tests, fix, check.
# Run `just` (or `just --list`) to see every available recipe.

# No Python needed — the entire ML pipeline runs in pure Rust.

default:
    @just --list

# ─── Setup ─────────────────────────────────────────────────────────────────

# One-shot setup: submodules + build.
setup: submodules build
    @echo ""
    @echo "Setup complete. Try: just run /path/to/video.mkv --dry-run"

# macOS only: install system tools (just, ffmpeg, git-lfs) via Homebrew.
brew:
    brew bundle

# Install development tools for formatting (taplo for TOML, shfmt for shell).
# Run this once to get all formatters.
install-fmt-tools:
    cargo install taplo-cli
    @# shfmt: install via Go or brew
    @if command -v brew >/dev/null 2>&1; then \
        brew install shfmt shellcheck 2>/dev/null || true; \
    fi
    @echo "✓ Formatting tools installed. Run 'just fix' to format all files."

# Initialise git submodules (vendor/ilass).
submodules:
    git submodule update --init --recursive

# ─── Build ─────────────────────────────────────────────────────────────────

# Build the release binary + the vendored ilass alignment engine.
build:
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

# ─── Tests ─────────────────────────────────────────────────────────────────

# Run the Rust test suite. Usage: `just test [extra cargo-test args]`
test *args:
    cargo test --workspace {{ args }}

# ─── Fix (auto-format all file types) ──────────────────────────────────────

# Fix all files in the repo: format Rust, TOML, shell scripts, Swift, JSON.
fix: fix-rust fix-toml fix-sh fix-swift fix-json
    @echo "✓ All files formatted."

# Format Rust code with rustfmt.
fix-rust:
    cargo +nightly fmt

# Format TOML files with taplo.
fix-toml:
    @if command -v taplo >/dev/null 2>&1; then \
        find . -name '*.toml' -not -path './.git/*' -not -path './target/*' -not -path './vendor/*' -exec taplo format {} +; \
    fi

# Format shell scripts with shfmt.
fix-sh:
    @if command -v shfmt >/dev/null 2>&1; then \
        find . -name '*.sh' -not -path './.git/*' -not -path './target/*' -not -path './vendor/*' -exec shfmt -w -i 2 {} +; \
    fi

# Format Swift files with swift-format (if available).
fix-swift:
    @if command -v swift-format >/dev/null 2>&1; then \
        find . -name '*.swift' -not -path './.git/*' -not -path './target/*' -not -path './vendor/*' -exec swift-format -i {} + 2>/dev/null || true; \
    fi

# Format JSON files with jq (canonical sort-keys formatting).
# Excludes .pi/ (pi-managed configs with intentional key ordering).
fix-json:
    @if command -v jq >/dev/null 2>&1; then \
        find . -name '*.json' -not -path './.git/*' -not -path './target/*' -not -path './vendor/*' -not -path './.pi/*' \
            -not -name 'package-lock.json' -not -name 'Cargo.lock' \
            -exec sh -c 'jq --sort-keys . "{}" > "{}.tmp" && mv "{}.tmp" "{}"' \; 2>/dev/null || true; \
    fi

# Auto-fix clippy warnings.
fix-clippy:
    cargo clippy --workspace --all-targets --fix --allow-dirty --allow-staged -- -D warnings

# ─── Check (validate formatting, no modifications) ─────────────────────────

# Check all file types for formatting issues. No modifications.
check: check-clippy check-fmt
    @echo "✓ All checks passed."

# Check Rust formatting (clippy + rustfmt).
check-clippy:
    cargo clippy --workspace --all-targets -- -D warnings

check-fmt: check-fmt-rust check-fmt-toml check-fmt-sh check-fmt-swift check-fmt-json
    @echo "✓ All formatting checks passed."

# Check Rust formatting with cargo fmt --check.
check-fmt-rust:
    cargo +nightly fmt --check

# Check TOML formatting with taplo.
check-fmt-toml:
    @if command -v taplo >/dev/null 2>&1; then \
        find . -name '*.toml' -not -path './.git/*' -not -path './target/*' -not -path './vendor/*' -exec taplo format --check {} + 2>&1 || { echo "TOML formatting issues found. Run: just fix-toml"; exit 1; }; \
    fi

# Check shell script formatting with shfmt --diff.
check-fmt-sh:
    @if command -v shfmt >/dev/null 2>&1; then \
        find . -name '*.sh' -not -path './.git/*' -not -path './target/*' -not -path './vendor/*' -exec shfmt -d -i 2 {} + 2>&1 || { echo "Shell formatting issues found. Run: just fix-sh"; exit 1; }; \
    fi

# Check Swift formatting (best-effort, no failure).
check-fmt-swift:
    @if command -v swift-format >/dev/null 2>&1; then \
        find . -name '*.swift' -not -path './.git/*' -not -path './target/*' -not -path './vendor/*' -exec swift-format -m format {} + 2>/dev/null || true; \
    fi

# Check JSON formatting with jq --sort-keys (reports diff failures).
# Excludes .pi/ (pi-managed configs with intentional key ordering).
check-fmt-json:
    @if command -v jq >/dev/null 2>&1; then \
        find . -name '*.json' -not -path './.git/*' -not -path './target/*' -not -path './vendor/*' -not -path './.pi/*' \
            -not -name 'package-lock.json' -not -name 'Cargo.lock' \
            -print0 | xargs -0 -I{} sh -c 'jq --sort-keys . "{}" | diff - "{}" >/dev/null 2>&1 || { echo "JSON formatting issue: {}"; exit 1; }'; \
    fi

# ─── Tidy (fix + lint + ordering) ──────────────────────────────────────────

# Full tidy: fix all formatting, clippy, and check dependency ordering.
tidy: fix fix-clippy tidy-check-deps
    @echo "✓ Tidy: all files formatted, linted, and dependencies sorted."

# Check that Cargo.toml [dependencies] entries are alphabetically sorted.
tidy-check-deps:
    bash scripts/check-deps-sorted.sh

# ─── Misc ──────────────────────────────────────────────────────────────────

# Install a git pre-commit hook that runs formatting fixes + checks.
install-hooks:
    @printf '#!/bin/sh\njust fix && just check && just tidy-check-deps\n' > .git/hooks/pre-commit
    @chmod +x .git/hooks/pre-commit
    @echo 'Pre-commit hook installed (runs just fix + just check + tidy-check-deps).'
