/**
 * Gate Extension — repo-specific auto-format + fast-gate enforcement.
 *
 * Ports the two Claude Code hooks into pi's event-driven extension model:
 *
 * 1. AUTO-FORMAT (was post-tool-use.sh): after every edit/write, runs the repo's
 *    own formatter on the touched file only. rustfmt for .rs, ruff for .py,
 *    shellcheck for .sh. Non-blocking — always succeeds, just formats.
 *
 * 2. FAST-GATE GUARD (was stop-gate.sh): when the agent finishes a turn and the
 *    working tree is dirty, runs the cheap formatter-check gates. If they fail,
 *    injects a "fix before claiming done" message for the next turn. The slow
 *    gates (clippy, cargo test, pytest) stay at manual `just check`/`just test`.
 */

import type {
	ExtensionAPI,
	ExtensionContext,
} from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { execSync } from "node:child_process";
import { existsSync } from "node:fs";

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
const RUFF = ".venv/bin/ruff";

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

/** Auto-format a single file with the repo's toolchain. Returns advisory text (unfixable lint) or empty. */
async function formatOneFile(pi: ExtensionAPI, fp: string): Promise<string> {
	let notes = "";

	if (fp.endsWith(".rs")) {
		const r = await pi.exec("rustfmt", ["--edition", "2024", fp]);
		if (r.code !== 0) notes = `rustfmt failed: ${r.stderr}`;
	} else if (fp.endsWith(".py")) {
		// ruff format (non-blocking)
		await pi.exec(RUFF, ["format", fp]);
		// ruff check --fix, surface unfixable
		const r = await pi.exec(RUFF, ["check", "--fix", fp]);
		if (r.code !== 0)
			notes = `ruff check (unfixable):\n${r.stdout || r.stderr}`;
	} else if (fp.endsWith(".sh")) {
		const r = await pi.exec("shellcheck", ["--severity=warning", fp]);
		if (r.code !== 0) notes = `shellcheck:\n${r.stdout || r.stderr}`;
	}
	return notes;
}

/** Run the fast-format-check gates (rustfmt --check + ruff check). Returns failure text or empty. */
async function runFastGates(pi: ExtensionAPI, dirty: string): Promise<string> {
	let fail = "";

	// Only run a gate if the relevant kind of file is dirty
	if (dirty.includes(".rs")) {
		const r = await pi.exec("cargo", ["fmt", "--check"]);
		if (r.code !== 0) {
			fail = `Rust formatting (run \`cargo fmt\` or \`just lint\`):\n${r.stdout || r.stderr}`;
		}
	}

	if (!fail && dirty.includes(".py")) {
		const r = await pi.exec(RUFF, ["check", "movie_translator/"]);
		if (r.code !== 0) {
			const lines = (r.stdout || r.stderr).split("\n");
			fail = `Python lint (run \`just lint\`):\n${lines.slice(-25).join("\n")}`;
		}
	}
	return fail;
}

export default function (pi: ExtensionAPI) {
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

		const notes = await formatOneFile(pi, fp);
		// Non-blocking: format already done. Advisory text available but not forced.
		if (notes) {
			// Inject as a custom message so the agent sees unfixable lint
			pi.sendMessage(
				{
					customType: "gate",
					content: `Auto-format applied to \`${fp}\`, but unfixable issues remain:\n\n\`\`\`\n${notes}\n\`\`\``,
					display: true,
				},
				{ deliverAs: "nextTurn" },
			);
		}
	});

	// ─── FAST-GATE GUARD on turn end ──────────────────────────────────────
	pi.on("turn_end", async (_event, _ctx) => {
		// Check for uncommitted changes
		const r = await pi.exec("git", ["status", "--porcelain"]);
		if (r.code !== 0) return; // not a git repo
		const dirty = r.stdout.trim();
		if (!dirty) return; // clean tree, nothing to gate

		const fail = await runFastGates(pi, dirty);
		if (fail) {
			pi.sendMessage(
				{
					customType: "gate",
					content: `## ⛔ Fast gate failed — fix before claiming done

${fail}

Fix it (\`just lint\` covers formatting + python lint), then run the full
\`just check && just test && just py-test\` before finishing.`,
					display: true,
				},
				{ deliverAs: "nextTurn" },
			);
		}
	});

	// ─── Register the /gate command for manual invocation ─────────────────
	pi.registerCommand("gate", {
		description: "Run the fast-format-check gates (rustfmt + ruff)",
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
					"Fast gate: PASSED ✓\n\nRemember: run the full `just check && just test && just py-test` before committing.",
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
			"Run the fast format-check gates (rustfmt --check + ruff check) to verify the repo is clean before claiming done. Does NOT run clippy, tests, or pytest — those are the slow gates.",
		promptSnippet: "Run rustfmt --check and ruff check to verify formatting",
		promptGuidelines: [
			"Use check_fast_gate before claiming work is complete to verify formatting and lint pass.",
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
							text: `⛔ Fast gate FAILED:\n\n${fail}\n\nFix with \`just lint\` then run the full \`just check && just test && just py-test\`.`,
						},
					],
					details: { clean: false, fail },
				};
			}
			return {
				content: [
					{
						type: "text",
						text: "✅ Fast gate PASSED (rustfmt check + ruff check).\n\nRemember: run the full `just check && just test && just py-test` before committing.",
					},
				],
				details: { clean: true },
			};
		},
	});
}
