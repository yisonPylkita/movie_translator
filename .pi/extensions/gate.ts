/**
 * Gate Extension — repo-specific auto-format + fast-gate enforcement.
 *
 * 1. AUTO-FORMAT: after every edit/write, runs `cargo +nightly fmt` on the
 *    touched .rs file. Non-blocking.
 *
 * 2. TOOLCHAIN GUARD: when rust-toolchain.toml is edited, sends a reminder to
 *    bump deliberately (it's the single source of truth for the compiler version).
 *
 * 3. FAST-GATE GUARD: when the agent finishes a turn with dirty tree, runs:
 *    - `cargo +nightly fmt --check`
 *    - `cargo sort -w --check`   (if Cargo.toml files are dirty)
 *    If any fail, injects a fix message.
 *
 * All formatting uses the nightly toolchain so the nightly-only
 * `group_imports = "StdExternalCrate"` in rustfmt.toml is applied.
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

/** Run all fast gates. Returns accumulated failure text (or empty on pass). */
async function runFastGates(pi: ExtensionAPI, dirty: string): Promise<string> {
	const failures: string[] = [];

	// Gate 1: Rust formatting (check .rs files)
	if (dirty.includes(".rs")) {
		const r = await pi.exec("cargo", ["+nightly", "fmt", "--check"]);
		if (r.code !== 0) {
			failures.push(
				`**Rust formatting** — run \`cargo +nightly fmt\` or \`just fix-rust\`:\n\`\`\`\n${r.stdout || r.stderr}\n\`\`\``,
			);
		}
	}

	// Gate 2: Cargo.toml dependency ordering (check Cargo.toml files)
	if (dirty.includes("Cargo.toml")) {
		const r = await pi.exec("cargo", ["sort", "-w", "--check"]);
		if (r.code !== 0) {
			failures.push(
				`**Cargo.toml dependency ordering** — run \`cargo sort -w\` or \`just fix\`:\n\`\`\`\n${r.stdout || r.stderr}\n\`\`\``,
			);
		}
	}

	return failures.join("\n\n");
}

export default function (pi: ExtensionAPI) {
	// Track gate failure state to avoid flooding repeated failures across turns.
	let lastGateFailed = false;

	// Track whether a toolchain change warning was already sent this session.
	let toolchainWarningSent = false;

	// ─── AUTO-FORMAT on edit/write ────────────────────────────────────────
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

		// ── AUTO-FORMAT for .rs files ─────────────────────────────────────
		if (!fp.endsWith(".rs")) return;

		// Use nightly toolchain so group_imports = StdExternalCrate is applied.
		const r = await pi.exec("cargo", ["+nightly", "fmt", "--", fp]);
		if (r.code !== 0) {
			pi.sendMessage(
				{
					customType: "gate",
					content: `Auto-format applied to \`${fp}\`, but rustfmt failed:\n\n\`\`\`\n${r.stderr}\n\`\`\``,
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

		const fail = await runFastGates(pi, dirty);
		if (fail && !lastGateFailed) {
			// Only send the failure message ONCE per failure cycle, not every turn.
			lastGateFailed = true;
			pi.sendMessage(
				{
					customType: "gate",
					content: `## ⛔ Fast gate failed — fix before claiming done

${fail}

Fix these issues (\`just fix\` covers formatting + deps sorting), then run
\`just check && just test\` before finishing.`,
					display: true,
				},
				{ deliverAs: "nextTurn" },
			);
		} else if (!fail) {
			lastGateFailed = false;
		}
	});

	// ─── Register the /gate command for manual invocation ─────────────────
	pi.registerCommand("gate", {
		description:
			"Run the fast gates (fmt + Cargo.toml sorting) on the dirty tree",
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
			const fail = await runFastGates(pi, dirty);
			if (fail) {
				ctx.ui.notify(`Fast gate FAILED:\n${fail}`, "error");
			} else {
				ctx.ui.notify(
					"Fast gate: PASSED ✓\n\nRemember: run the full `just check && just test` before committing.",
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
			"Run the fast gates (cargo +nightly fmt --check + cargo sort -w --check if applicable) to verify the repo is clean. Does NOT run clippy, tests — those are the slow gates.",
		promptSnippet:
			"Run cargo +nightly fmt --check + cargo sort -w --check to verify formatting and deps ordering",
		promptGuidelines: [
			"Use check_fast_gate before claiming work is complete to verify formatting and dependency ordering pass.",
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
			const fail = await runFastGates(pi, dirty);
			if (fail) {
				return {
					content: [
						{
							type: "text",
							text: `⛔ Fast gate FAILED:\n\n${fail}\n\nFix with \`just fix\` then run \`just check && just test\`.`,
						},
					],
					details: { clean: false, fail },
				};
			}
			return {
				content: [
					{
						type: "text",
						text: "✅ Fast gate PASSED (fmt + deps).\n\nRemember: run `just check && just test` before committing.",
					},
				],
				details: { clean: true },
			};
		},
	});
}
