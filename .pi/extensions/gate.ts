/**
 * Gate Extension — repo-specific auto-format + fast-gate enforcement.
 *
 * 1. AUTO-FIX: after every edit/write of a tracked file, runs `just fix-rust`
 *    (formats .rs files with `cargo +nightly fmt`) so the nightly-only
 *    `group_imports = "StdExternalCrate"` setting is always applied.
 *
 * 2. TOOLCHAIN GUARD: when rust-toolchain.toml is edited, sends a reminder to
 *    bump deliberately (it's the single source of truth for the compiler version).
 *
 * 3. FAST-GATE: when the agent finishes a turn with dirty tree, runs
 *    `just check-fmt-rust` to verify formatting. If it fails, injects a
 *    fix-next-turn message.
 *
 * All formatting uses the nightly toolchain via the justfile recipes so the
 * nightly-only `group_imports = "StdExternalCrate"` in rustfmt.toml is applied.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { execSync } from "node:child_process";

/** Resolve the git repo root at extension load time (once). */
function resolveRepoRoot(): string {
	try {
		return execSync("git rev-parse --show-toplevel", {
			encoding: "utf-8",
		}).trim();
	} catch {
		return process.cwd();
	}
}
const REPO_ROOT = resolveRepoRoot();

/** True if path is inside the repo and not in a vendored/generated tree. */
function isTrackedFile(path: string): boolean {
	if (!path.startsWith(REPO_ROOT + "/")) return false;
	const rel = path.slice(REPO_ROOT.length + 1);
	if (
		rel.startsWith("vendor/") ||
		rel.startsWith("target/") ||
		rel.startsWith(".venv/")
	)
		return false;
	return true;
}

export default function (pi: ExtensionAPI) {
	// Track gate failure state to avoid flooding repeated failures across turns.
	let lastGateFailed = false;

	// Track whether a toolchain change warning was already sent this session.
	let toolchainWarningSent = false;

	// ─── AUTO-FIX on edit/write ──────────────────────────────────────────
	pi.on("tool_result", async (event, _ctx) => {
		if (event.toolName !== "edit" && event.toolName !== "write") return;
		if (event.isError) return;

		// Try to extract the file path from tool input
		const input = event.input as Record<string, unknown> | undefined;
		let fp: string | undefined;
		if (input?.path && typeof input.path === "string") fp = input.path;
		else if (input?.filePath && typeof input.filePath === "string")
			fp = input.filePath;
		if (!fp || !isTrackedFile(fp)) return;

		// ── TOOLCHAIN GUARD ───────────────────────────────────────────────
		if (
			!toolchainWarningSent &&
			(fp.endsWith("rust-toolchain.toml") || fp.endsWith("rust-toolchain"))
		) {
			toolchainWarningSent = true;
			pi.sendMessage(
				{
					customType: "gate",
					content: `## ⚠️ Toolchain version changed

\`${fp}\` was modified. **\`rust-toolchain.toml\` is the single source of truth**
for the Rust compiler version (local + CI both read it via \`rustup show\`).

Before claiming done:
1. Confirm the bump is intentional
2. Update \`.github/workflows/tests.yml\` if it has a separate toolchain pin
3. Update \`ONBOARDING.md\` / \`README.md\` if they mention minimum versions

Bump deliberately.`,
					display: true,
				},
				{ deliverAs: "nextTurn" },
			);
		}

		// ── AUTO-FIX for .rs files ───────────────────────────────────────
		if (!fp.endsWith(".rs")) return;

		// Delegate to `just fix-rust` (cargo +nightly fmt) so the nightly-only
		// group_imports = StdExternalCrate setting is applied across the workspace.
		const r = await pi.exec("just", ["fix-rust"]);
		if (r.code !== 0) {
			pi.sendMessage(
				{
					customType: "gate",
					content: `Auto-fix failed for \`${fp}\`:\n\n\`\`\`\n${r.stderr}\n\`\`\``,
					display: true,
				},
				{ deliverAs: "nextTurn" },
			);
		}
	});

	// ─── FAST-GATE GUARD on turn end ──────────────────────────────────────
	pi.on("turn_end", async (_event, _ctx) => {
		const r = await pi.exec("git", ["status", "--porcelain"]);
		if (r.code !== 0) return;
		const dirty = r.stdout.trim();

		if (!dirty) {
			// Tree is clean — reset gate state.
			lastGateFailed = false;
			return;
		}
		if (!dirty.includes(".rs")) return;

		// Delegate to `just check-fmt-rust` (cargo +nightly fmt --check)
		// instead of re-implementing a subset of checks.
		const fmt = await pi.exec("just", ["check-fmt-rust"]);
		if (fmt.code !== 0 && !lastGateFailed) {
			lastGateFailed = true;
			pi.sendMessage(
				{
					customType: "gate",
					content: `## ⛔ Rust formatting gate failed — fix before claiming done

\`\`\`
${fmt.stdout || fmt.stderr}
\`\`\`

Run \`just fix-rust\` to fix, then \`just check && just test\` to finish.`,
					display: true,
				},
				{ deliverAs: "nextTurn" },
			);
		} else if (fmt.code === 0) {
			lastGateFailed = false;
		}
	});

	// ─── Register the /gate command for manual invocation ─────────────────
	pi.registerCommand("gate", {
		description: "Run just check-fmt-rust on the dirty tree",
		handler: async (_args, ctx) => {
			ctx.ui.notify("Running fast gates...", "info");
			const r = await pi.exec("git", ["status", "--porcelain"]);
			if (r.code !== 0) {
				ctx.ui.notify("Not a git repository", "error");
				return;
			}
			const dirty = r.stdout.trim();
			if (!dirty) {
				ctx.ui.notify("Working tree clean — nothing to gate", "info");
				return;
			}
			const fmt = await pi.exec("just", ["check-fmt-rust"]);
			if (fmt.code !== 0) {
				ctx.ui.notify(
					`Formatting check FAILED:\n${fmt.stdout || fmt.stderr}`,
					"error",
				);
			} else {
				ctx.ui.notify(
					"Formatting: PASSED ✓\n\nRemember: run the full `just check && just test` before committing.",
					"info",
				);
			}
		},
	});

	// Register a tool the LLM can call to check the fast gates
	pi.registerTool({
		name: "check_fast_gate",
		label: "Check Fast Gate",
		description:
			"Run just check-fmt-rust (cargo +nightly fmt --check) to verify Rust formatting is clean. Does NOT run clippy or tests — those are the slow gates.",
		promptSnippet: "Run just check-fmt-rust to verify formatting",
		promptGuidelines: [
			"Use check_fast_gate before claiming work is complete to verify formatting passes.",
		],
		parameters: Type.Object({}),
		async execute(_toolCallId, _params, _signal, _onUpdate, _ctx) {
			const r = await pi.exec("git", ["status", "--porcelain"]);
			const dirty = r.stdout.trim();
			if (!dirty) {
				return {
					content: [
						{ type: "text", text: "✅ Working tree clean — nothing to gate." },
					],
					details: { clean: true },
				};
			}
			const fmt = await pi.exec("just", ["check-fmt-rust"]);
			if (fmt.code !== 0) {
				return {
					content: [
						{
							type: "text",
							text: `⛔ Rust formatting FAILED:\n\n\`\`\`\n${fmt.stdout || fmt.stderr}\n\`\`\`\n\nFix with \`just fix-rust\` then run \`just check && just test\`.`,
						},
					],
					details: { clean: false, fail: fmt.stdout || fmt.stderr },
				};
			}
			return {
				content: [
					{
						type: "text",
						text: "✅ Formatting: PASSED.\n\nRemember: run `just check && just test` before committing.",
					},
				],
				details: { clean: true },
			};
		},
	});
}
