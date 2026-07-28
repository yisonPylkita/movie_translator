# Pi harness cost policy

**Status:** Living. Enforced by `AGENTS.md`, `.pi/agents/*`, and `.pi/settings.json`.

**Goal:** lead model spends tokens on judgment; cheaper children handle bounded scans, implementation, and long checks.

## Model ladder

| Role | Project default | Thinking | Work |
| --- | --- | --- | --- |
| Session lead | Owner-selected | as needed | Decompose, decide, steer, accept, report |
| `stage_owner` | `deepseek/deepseek-v4-pro` | medium | One bounded package through writer/review/verify |
| `bounded_worker` / builtin `worker` | `deepseek/deepseek-v4-flash` | medium | Exclusive-scope implementation and checks |
| `fast_explorer` / scout | `deepseek/deepseek-v4-flash` | low | Read-only repository evidence |
| `spec_analyst` | `deepseek/deepseek-v4-flash` | medium | Read-only design and invariant analysis |
| builtin `oracle` | `deepseek/deepseek-v4-pro` | medium | Hard decision-consistency review |

Override per run only when risk or complexity needs stronger model.

## Lead discipline

Prefer delegating:

- full `just check && just test` runs;
- long cargo suites;
- broad mechanical edits;
- repeated repository inventory;
- commit/push when user explicitly requested it.

Lead may run focused reads and cheap checks needed to supervise safely. Never sleep-poll subagents. Use status/steer/wait controls. Never duplicate credible green full-gate evidence without reason.

## Work-package shape

Every writer/stage contract includes:

- expected baseline and dirty-tree constraints;
- exclusive write paths and read-only shared paths;
- one outcome and explicit non-goals;
- pure-Rust, GPU serialization, platform, provider, and destructive-action boundaries;
- focused checks and at most one full gate;
- commit/push permission;
- stop condition and handoff schema.

Prefer serial single-writer packages over mega-prompts. Maximum three concurrent children; one writer per worktree.

## Expensive and outward validation

Default to source/tests, `--dry-run`, or tiny synthetic media. Real translation/OCR/ASR/inpaint runs consume GPU and may compile Apple bridges. Provider fetches and browser workflows contact external systems. `--in-place` is destructive. Agent task must explicitly authorize these.

## Owner controls

- Ensure pinned providers/models are available before long workflows.
- Change project model pins deliberately in `.pi/settings.json` and matching custom agents.
- Keep watchdog off for cost-sensitive work unless adversarial review value warrants it.
