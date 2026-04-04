# Movie Translator development tasks

default:
    @just --list

# Run linter and formatter (auto-fix)
lint:
    uv run ruff check --fix .
    uv run ruff format .

# Run all checks without modifying files (mirrors CI)
check:
    uv run ruff check .
    uv run ruff format --check .
    uv run ty check --error-on-warning

# Run tests
test *args:
    uv run pytest -v {{ args }}

# Run all checks and tests (CI equivalent)
ci: check test

# Run movie-translator CLI
run dir *args:
    uv run movie-translator "{{ dir }}" {{ args }}

# Install git pre-commit hook
install-hooks:
    @echo '#!/bin/sh' > .git/hooks/pre-commit
    @echo 'just check' >> .git/hooks/pre-commit
    @chmod +x .git/hooks/pre-commit
    @echo 'Pre-commit hook installed (runs just check).'
