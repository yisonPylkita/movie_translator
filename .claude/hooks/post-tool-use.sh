#!/usr/bin/env bash
#
# PostToolUse hook (matcher: Edit|Write|MultiEdit). Auto-formats the file the
# agent just touched with the repo's own formatters, and surfaces syntax/lint
# errors back to the agent. Non-blocking by design — ALWAYS exits 0. The value
# is the in-place format (deterministic, can't be skipped), not the advisory
# text (some Claude Code versions drop PostToolUse additionalContext).
#
# Fast: operates on the single touched file only — never `cargo fmt` over the
# whole workspace, never the whole package. rustfmt on one file; ruff on one
# file. The slow gates (clippy, cargo test, pytest, ty) stay at `just check` /
# `just test` / `just py-test` / CI.
set -uo pipefail

HERE="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$HERE" || exit 0

# The edited file path comes in on stdin as part of Claude Code's hook JSON.
fp="$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))' 2>/dev/null)" || exit 0
[ -n "$fp" ] && [ -f "$fp" ] || exit 0
# Only ever touch files inside this repo.
case "$fp" in "$HERE"/*) ;; *) exit 0 ;; esac
# Never touch vendored / generated trees.
case "$fp" in "$HERE"/vendor/* | "$HERE"/target/* | "$HERE"/.venv/*) exit 0 ;; esac

RUFF=".venv/bin/ruff"
notes=""

case "$fp" in
*.rs)
  # rustfmt is installed as a toolchain component (rust-toolchain.toml). Edition
  # matches the workspace (2021). Best-effort; if it rewrites, great.
  if command -v rustfmt >/dev/null 2>&1; then
    err="$(rustfmt --edition 2021 "$fp" 2>&1)" || notes="rustfmt failed (likely a syntax error): $err"
  fi
  ;;
*.py)
  if [ -x "$RUFF" ]; then
    "$RUFF" format "$fp" >/dev/null 2>&1 || true
    # --fix auto-resolves what it can; surface what it can't.
    err="$("$RUFF" check --fix "$fp" 2>&1)" || notes="ruff check (unfixable): $err"
  fi
  ;;
*.sh)
  if command -v shellcheck >/dev/null 2>&1; then
    err="$(shellcheck --severity=warning "$fp" 2>&1)" || notes="shellcheck: $err"
  fi
  ;;
esac

# If a syntax/lint problem the formatter can't auto-fix remains, advise the
# agent (best-effort; the auto-format above already happened regardless).
if [ -n "$notes" ]; then
  python3 -c 'import json,sys; print(json.dumps({"hookSpecificOutput":{"hookEventName":"PostToolUse","additionalContext":sys.argv[1]}}))' "$notes" 2>/dev/null || true
fi
exit 0
