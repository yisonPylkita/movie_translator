#!/bin/bash
# Check import hygiene using ast-grep.
set -euo pipefail

RULE="${1:-.pi/rules/ast-grep-rules/rules/import-function-over-path.yml}"
TARGET="${2:-crates/}"

if ! command -v sg &>/dev/null; then
	echo "  Import hygiene: ast-grep (sg) not found — install: brew install ast-grep"
	exit 0
fi

json=$(sg scan --rule "$RULE" "$TARGET" --json 2>/dev/null || echo "[]")
count=$(echo "$json" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(len(d))' 2>/dev/null || echo "?")

echo "  Import hygiene: $count fully-qualified calls found (VERBOSE=1 to list)"

if [ "${VERBOSE:-0}" = "1" ] && [ "$count" -gt 0 ] 2>/dev/null; then
	sg scan --rule "$RULE" "$TARGET"
fi
