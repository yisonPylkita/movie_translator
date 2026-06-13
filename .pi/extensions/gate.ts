/**
 * Gate Extension — repo-specific auto-format + fast-gate enforcement.
 *
 * 1. AUTO-FORMAT: after every edit/write, runs `cargo +nightly fmt` on the
 *    touched .rs file. Non-blocking.
 *
 * 2. FAST-GATE GUARD: when the agent finishes a turn with dirty tree, runs
 *    `cargo +nightly fmt --check`. If it fails, injects a fix message.
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

/** Run the fast-format-check gates (cargo +nightly fmt --check). Returns failure text or empty. */
async function runFastGates(pi: ExtensionAPI, dirty: string): Promise<string> {
	if (!dirty.includes(".rs")) return "";

	const r = await pi.exec("cargo", ["+nightly", "fmt", "--check"]);
	if (r.code !== 0) {
		return `Rust formatting (run \`cargo +nightly fmt\` or \`just lint\`):\n${r.stdout || r.stderr}`;
	}
	return "";
}

export default function (pi: ExtensionAPI) {
	// Track gate failure state to avoid flooding repeated failures across turns.
	let lastGateFailed = false;

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
		if (!fp || !isTrackedFile(fp) || !fp.endsWith(".rs")) return;

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

Fix it (\`just lint\` covers formatting), then run the full
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
		description: "Run the fast-format-check gate (cargo +nightly fmt --check)",
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
			"Run the fast format-check gate (cargo +nightly fmt --check) to verify the repo is clean. Does NOT run clippy, tests — those are the slow gates.",
		promptSnippet: "Run cargo +nightly fmt --check to verify formatting",
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
			const fail = await runFastGates(pi, dirty);
			if (fail) {
				return {
					content: [
						{
							type: "text",
							text: `⛔ Fast gate FAILED:\n\n${fail}\n\nFix with \`just lint\` then run \`just check && just test\`.`,
						},
					],
					details: { clean: false, fail },
				};
			}
			return {
				content: [
					{
						type: "text",
						text: "✅ Fast gate PASSED (cargo +nightly fmt --check).\n\nRemember: run `just check && just test` before committing.",
					},
				],
				details: { clean: true },
			};
		},
	});
}
