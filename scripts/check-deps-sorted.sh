#!/usr/bin/env bash
# Check that each Cargo.toml's [dependencies] has external deps first,
# a blank line, then internal workspace deps — each group sorted alphabetically.
# Internal workspace crates are identified by the keys listed in
# INTERNAL_CRATES below.
# Exits non-zero on any violation.

set -euo pipefail

# Workspace crate names (prefix-matches, so mt-core matches mt-core, mt-fetch, etc.)
INTERNAL_CRATES="^mt-"

fail=0

for f in Cargo.toml crates/*/Cargo.toml; do
	[ -f "$f" ] || continue
	in_section=0
	unsorted=0
	saw_internal=0
	prev_ext=""
	prev_int=""

	while IFS= read -r line; do
		# Section detection
		if [[ "$line" =~ ^\[(dependencies|build-dependencies|dev-dependencies) ]]; then
			in_section=1
			prev_ext=""
			prev_int=""
			saw_internal=0
			continue
		fi
		[[ "$line" =~ ^\[.*\]$ ]] && in_section=0
		[ "$in_section" -eq 0 ] && continue

		# Skip blanks and comments
		[[ "$line" =~ ^[[:space:]]*$ ]] && continue
		[[ "$line" =~ ^[[:space:]]*# ]] && continue

		# Extract key before =, {, or space
		key="$(echo "$line" | sed 's/^[[:space:]]*//;s/[={}].*//;s/[[:space:]]*$//')"
		[ -z "$key" ] && continue

		if echo "$key" | grep -qE "$INTERNAL_CRATES"; then
			# Internal workspace dependency
			saw_internal=1
			if [ -n "$prev_int" ] && [ "$key" \< "$prev_int" ]; then
				echo "UNSORTED (internal) in $f: '$prev_int' before '$key'"
				unsorted=1
			fi
			prev_int="$key"
		else
			# External dependency
			if [ "$saw_internal" -eq 1 ]; then
				echo "ORDER ERROR in $f: external dep '$key' appears after internal deps"
				unsorted=1
			fi
			if [ -n "$prev_ext" ] && [ "$key" \< "$prev_ext" ]; then
				echo "UNSORTED (external) in $f: '$prev_ext' before '$key'"
				unsorted=1
			fi
			prev_ext="$key"
		fi
	done <"$f"

	if [ "$unsorted" -ne 0 ]; then
		fail=1
	fi
done

if [ "$fail" -ne 0 ]; then
	echo ""
	echo "Cargo.toml dependency sections must be:"
	echo "  1. External (crates.io) deps — alphabetically sorted"
	echo "  2. Blank line"
	echo "  3. Internal (mt-*) workspace deps — alphabetically sorted"
	exit 1
fi

echo "✓ All Cargo.toml dependencies are correctly structured and sorted."
