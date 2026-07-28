# Agent orchestration and work-package protocol

**Status:** Living  
**System:** movie-translator

## Purpose

Controlled AI-assisted implementation: bounded responsibility, stable crate contracts, conflict-free writes, independent evidence, and coherent final reports.

```text
User goal
  → lead scopes work package
  → read-only evidence/review fanout
  → one exclusive-scope writer
  → focused verification
  → independent review when warranted
  → one correction writer
  → full gate and parent acceptance
```

Repository remains source of truth. Chat and agent output never replace code, tests, current docs, or observed command results.

## Principles

1. Lead coordinates and accepts; does not duplicate every worker action.
2. Agents receive one bounded work package, not “fix everything.”
3. Parallelism requires disjoint files or isolated worktrees.
4. One active writer per worktree.
5. Ambiguous product, architecture, destructive, external, privacy, or expensive-run choices escalate.
6. Completion reports identify changes, evidence, unresolved issues, and integration impact.
7. Broad/cross-crate/risky work gets independent review.
8. Checks not run are reported as not run, never pass.

## Roles

| Role | Responsibility |
| --- | --- |
| Lead | Scope, dependency/file ownership, decisions, synthesis, acceptance |
| `fast_explorer` | Read-only current-code evidence |
| `spec_analyst` | Read-only design, invariant, platform, benchmark, acceptance analysis |
| `bounded_worker` | One exclusive write scope plus focused checks |
| `stage_owner` | One broader work package through writer, review, and verification |
| Reviewer | Independent correctness, regression, test, simplicity, or domain review |
| `gate-verify` skill | Full `just check` and `just test` evidence |
| Domain skills | ML-stage, subtitle-fetch/alignment, and benchmark diagnostics |

## Universal constraints

- Pure Rust Cargo workspace; zero Python/PyO3/venv dependency.
- Translation/OCR/inpaint jobs remain serialized through one GPU worker.
- GPU executor calls remain off async runtime workers via blocking boundary.
- Subtitle dialogue classification stays structural, not keyword driven.
- Every viable fetched Polish candidate at/above threshold remains separate output track.
- Provider calls use rate limiting and backoff.
- Apple Translation, Vision, and Speech integrations remain platform guarded.
- Import functions/macros/types and use short names per `AGENTS.md`.
- No unnecessary explicit type annotation where inference is obvious.
- No real providers, browser resolver, GPU/media run, `--in-place`, or other outward/destructive action without explicit authorization.
- No force, rebase, `reset --hard`, or amend-after-push.

## Work-package task envelope

Use compact Markdown or YAML. Preserve exact paths and symbols.

```yaml
agent_task:
  id: "MT-<area>-<number>"
  goal: "One observable outcome"
  baseline:
    branch: "current branch"
    dirty_tree: "list known pre-existing paths"
  scope:
    write:
      - "exclusive/path/**"
    read:
      - "relevant/dependency.rs"
    forbidden:
      - "unrelated/**"
  acceptance:
    - "behavior or test expectation"
  constraints:
    - "applicable universal/domain invariants"
  verification:
    focused:
      - "cargo test -p mt-... <name>"
    full_gate: "just check && just test | not required with reason"
  git:
    commit: false
    push: false
  stop_conditions:
    - "shared contract change not granted"
    - "dirty overlap"
    - "real external/GPU/destructive action needed"
```

Shared files require explicit ownership: root manifests/lockfile, `justfile`, CI/toolchain, `README.md`, `AGENTS.md`, architecture/design docs, and foundational public APIs used across crates.

## Writer prompt

```markdown
You are exclusive-scope writer for attached task.

Read `AGENTS.md`, assigned files, direct callers, and relevant tests before editing. Stop on dirty overlap or need for unnamed shared-file changes. Preserve task constraints. Make smallest coherent change; no unrelated cleanup.

Run focused checks first. Run full gate only when task requests it; do not repeat a green full gate. Always run `git diff --check`. Do not commit or push unless task grants it.

Return changed paths, behavior, commands/results, assumptions, residual risks, and unmade decisions.
```

## Read-only review prompts

### Correctness and regression

```markdown
Review actual diff and affected callers/tests. Find concrete correctness, regression, concurrency, error-path, and platform issues. Check movie-translator invariants from `AGENTS.md`. Do not edit. Report only evidence-backed findings with severity, path/line, impact, and smallest safe fix. State no findings when clean.
```

### Subtitle pipeline

```markdown
Review fetch/validation/alignment changes. Check structural dialogue classification, timing offsets, ilass/xcorr fallback, candidate threshold semantics, multi-track retention, provider rate limiting, and malformed subtitle handling. Do not edit. Cite paths and tests.
```

### ML/GPU/platform

```markdown
Review translation/OCR/ASR/inpaint changes. Check one-worker serialization, blocking boundaries, Swift bridge compilation/error reporting, macOS availability guards, unsupported-platform behavior, artifact cleanup, and accidental real GPU/media execution in tests. Do not edit.
```

### Validation quality

```markdown
Review tests and command evidence against stated acceptance. Distinguish focused checks, full gate, dry runs, synthetic media, and real external/GPU execution. Flag untested changed paths, false pass claims, flaky external dependencies, and benchmark-sensitive behavior lacking evidence. Do not edit.
```

## Result envelope

```yaml
agent_result:
  task_id: "MT-<area>-<number>"
  status: "complete | partial | blocked | failed"
  files:
    added: []
    modified: []
    deleted: []
  behavior:
    implemented: []
    deliberately_unchanged: []
  checks:
    - command: "cargo test -p mt-..."
      result: "pass | fail | skipped"
      evidence: "concise output"
  reviews:
    findings_fixed: []
    findings_deferred: []
  blockers: []
  risks: []
  git:
    commit: null
    push: "not requested"
```

## Blocker protocol

Classify blocker explicitly:

- `DEPENDENCY_MISSING`
- `CONTRACT_AMBIGUITY`
- `DESIGN_CONFLICT`
- `PLATFORM_UNAVAILABLE`
- `GPU_OR_MODEL_UNAVAILABLE`
- `EXTERNAL_PROVIDER_UNAVAILABLE`
- `FILE_OWNERSHIP_CONFLICT`
- `VERIFICATION_UNAVAILABLE`
- `DESTRUCTIVE_ACTION_REQUIRES_APPROVAL`
- `PRIVACY_OR_CREDENTIAL_SCOPE`

Report exact question, evidence, affected paths, safe partial work, and recommended resolution. Do not bury blocker in notes or guess through it.

## Parallelism and integration

Safe parallel lanes:

- read-only code evidence plus read-only design analysis;
- Rust crate work versus tooling/docs only when files and shared contracts do not overlap;
- independent reviewers with distinct angles;
- isolated worktrees for intentional concurrent writers on clean baseline.

Unsafe:

- two writers on same worktree/file;
- parallel GPU jobs;
- concurrent root manifest/lockfile changes;
- repeated provider fetches;
- parent edits while async child writes same worktree.

Integration order:

1. foundational `mt-core` contracts;
2. subtitle parsing/model changes;
3. discovery/fetch/media/ML producers;
4. pipeline orchestration;
5. CLI/TUI;
6. tests/benchmarks/docs/tooling.

Narrow mechanical conflicts may be resolved directly. Semantic cross-crate contract conflicts return to lead.

## Verification ladder

Spend cheapest adequate evidence first:

1. formatter/LSP/static diagnostics;
2. focused crate/unit tests;
3. synthetic or dry-run path;
4. `just check`;
5. `just test`;
6. benchmark, real media, GPU, provider, or browser run only when change requires and owner authorizes it.

Observe produced artifact when behavior matters; exit code alone may be insufficient. Never regenerate expected benchmark/golden output only to turn failure green.

## Acceptance rules

Package accepted when:

- scope matches assignment;
- no unrelated user changes were overwritten;
- acceptance behavior has evidence;
- blocking review findings are fixed or explicitly escalated;
- required checks pass;
- `git diff --check` is clean;
- full gate passes when required;
- remaining risks and unrun checks are explicit.

Commit/push only inside user-approved boundary.
