#!/usr/bin/env bash
#
# Stop hook: before the agent ends its turn, enforce the repo's #1 rule —
# "gate before done" — deterministically, instead of trusting prose the model
# can skip. Runs only the FAST gates (format checks), and only when there are
# uncommitted changes, so pure-conversation turns aren't slowed.
#
# Fast gate = `cargo fmt --check` (rustfmt only, no compile) + `ruff check`
# (python lint). The SLOW gates stay where they belong:
#   clippy  -> `just check`        (full `cargo clippy -D warnings`, can be minutes)
#   tests   -> `just test`         (`cargo test --workspace`)
#   py-test -> `just py-test`      (`pytest`)
#   CI      -> all of the above on Linux + macOS
# Blocks the turn (exit 2) on a fast-gate failure, feeding the reason back so
# the agent fixes it rather than claiming success.
set -uo pipefail

HERE="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$HERE" || exit 0

# Loop guard: if we're already inside a Stop-hook continuation, don't recurse.
active="$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("stop_hook_active", False))' 2>/dev/null)" || active="False"
[ "$active" = "True" ] && exit 0

# Nothing to gate if the working tree is clean.
[ -n "$(git status --porcelain 2>/dev/null)" ] || exit 0

# Only run a gate if the relevant kind of file is dirty — keep the hook cheap.
dirty="$(git status --porcelain 2>/dev/null)"
fail=""

if printf '%s\n' "$dirty" | grep -qE '\.rs$'; then
  if ! out="$(cargo fmt --check 2>&1)"; then
    fail="Rust formatting (run \`cargo fmt\` or \`just lint\`):
$out"
  fi
fi

if [ -z "$fail" ] && printf '%s\n' "$dirty" | grep -qE '\.py$'; then
  if [ -x ".venv/bin/ruff" ] && ! out="$(.venv/bin/ruff check movie_translator/ 2>&1)"; then
    fail="Python lint (run \`just lint\`):
$(printf '%s\n' "$out" | tail -25)"
  fi
fi

if [ -n "$fail" ]; then
  printf 'Stop blocked — the fast gate is not green:\n\n%s\n\nFix it (or `just lint`), then finish the turn. Run the full `just check && just test && just py-test` before claiming done.\n' "$fail" >&2
  exit 2
fi
exit 0
