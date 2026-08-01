# Pi harness cost policy

**Status:** Living. Enforced by `AGENTS.md`, `.pi/agents/*`, and `.pi/settings.json`.

**Goal:** lead model spends tokens on judgment; cheaper children handle bounded scans, implementation, and long checks.

## Model ladder

| Role | Project default | Thinking | Work |
| --- | --- | --- | --- |
| Session lead | Owner-selected | as needed | Decompose, decide, steer, accept, report |
| `stage_owner` | `deepseek/deepseek-v4-flash` | low | One bounded package through writer/review/verify |
| `bounded_worker` / builtin `worker` | `deepseek/deepseek-v4-flash` | medium | Exclusive-scope implementation and checks |
| `fast_explorer` / scout | `deepseek/deepseek-v4-flash` | low | Read-only repository evidence |
| `spec_analyst` | `deepseek/deepseek-v4-flash` | medium | Read-only design and invariant analysis |
| builtin `oracle` | `deepseek/deepseek-v4-flash` | low | Hard decision-consistency review |

Pro (`deepseek/deepseek-v4-pro`) allowed for explicit per-call escalation only when risk or complexity requires stronger model.

## Lead discipline

Trivial/small (~1–2-file) work: lead handles directly. No mandatory delegation.
Long/mechanical work, repeated inventory, full gate runs: delegate to workers.

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

## Workflow policy (v0.5.0)

- **Packaging.** Split broad work into sequential bounded packages (foundation, engine,
  product surface or equivalent). Max 3 concurrent children: 1 writer + up to 2 read-only,
  only when useful. Packages needing >3 lanes split into sequential packages. Fresh owner
  context + compact artifact handoff per package. Parent receives ONE final synthesis.
- **Discovery.** Default one scout; second only for a genuinely independent domain; skip
  scouting when brief/handoff already supplies evidence. No three-scout fanout.
- **Dispatch.** Exactly one subagent invocation per owner turn; concurrent children in ONE
  parallel call; never issue while a call is active. Persist child run IDs + named artifacts
  in ledger; use status by ID; never reconstruct work via repo archaeology.
- **Writer safety.** One writer per cwd/worktree; parallel writers only in isolated clean
  git worktrees; parallel read-only fine; Cargo/build/full gates in same cwd serialize.
- **Token discipline.** Child handoffs ≤600 tokens; owner ≤1000. Logs to named artifacts
  (`.pi/tool-output/`) file-only. Never set persistent toolBudget/turnBudget.
- **Gates.** Targeted tests per lane; one full `just check && just test` final gate per
  coherent delivery owned by the gate tester; reviewer read-only, never reruns gates;
  baseline full gate only evidence-driven; no repeat after already-green.
- **Review cadence.** Review engine/process/filesystem/security slices before >~1500-2000
  net new core lines; resolve blocker/HIGH before dependent lane; verdicts
  `accepted` | `blocking`; no approve with unresolved HIGH.
- **Dirty worktree.** Preserve unrelated staged/user work; path-scoped evidence;
  pre-existing staged files never cause rejection; default no commit/push.
- **Ledger.** Owner keeps compact named artifact ledger per package (package, writer,
  changed paths, focused gate, review verdict, run/output ID, pending decisions).
  Checkpoint per package without dumping transcripts.

Port workflow-policy deltas into project configs after active runs complete; do not edit
project configs mid-run.

## Expensive and outward validation

Default to source/tests, `--dry-run`, or tiny synthetic media. Real translation/OCR/ASR/inpaint runs consume GPU and may compile Apple bridges. Provider fetches and browser workflows contact external systems. `--in-place` is destructive. Agent task must explicitly authorize these.

## Owner controls

- Ensure pinned providers/models are available before long workflows.
- Change project model pins deliberately in `.pi/settings.json` and matching custom agents.
- Keep watchdog off for cost-sensitive work unless adversarial review value warrants it.
- Never set persistent `toolBudget`/`turnBudget` on Flash agents in config or ordinary call
  sites; caller-lifetime budget only for exceptional bounded diagnostics.
