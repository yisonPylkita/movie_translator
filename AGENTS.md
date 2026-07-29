# AGENTS.md — movie_translator

One-shot orientation. English→Polish video subtitle translator (MKV/MP4).

**Detailed project guide:** load `.pi/PROJECT_GUIDE.md` on demand for architecture, commands, invariants, import hygiene, output hygiene, handoff format, project memory files, and domain skills.

## Agentic workflow & delegation contract

### Hierarchy

- **Parent (GPT-5.6 solo).** Product discussion, discovery, and trivial-to-small (~1–2-file) work directly. May spawn one focused Flash leaf for narrow tasks. No mandatory Pro hop.
- **Initiative Owner (`stage_owner`, DeepSeek v4 Flash).** Reserved for broad, cross-cutting, or risky packages. Sole orchestrator per package. Validates whole result, returns ≤1000-token handoff. Spawns Flash leaf children. Never edits files directly. Escalates only genuine product/architecture/destructive/outward ambiguity.
- **Leaf Children (DeepSeek v4 Flash).** Execute exclusive scope. Never delegate
  recursively. Contact owner for ambiguity.

### Rules

- **Skill loading:** Children load named domain skills (`.pi/skills/<name>/SKILL.md`)
  on demand. Parent never reads full SKILL.md — only references catalog.
- **Parallelise by disjoint file lanes:** Rust crates (`crates/**`) + tooling/docs
  (`justfile`, `.github/**`, `docs/**`). Never same-file.
- **Serialize GPU + outward actions.** `--in-place`, `git lfs`, looped provider
  fetches — confirm, don't parallelise.
- **Verify via gate chain, cite evidence.** Never assert "done" without
  `just check` / `just test` output.
- **No code index** — grep/ripgrep. Prefer compiler-driven refactors.
- **One writer per worktree.** Max 3 concurrent children. Shared manifests, lockfiles,
  CI/toolchain, README.md, AGENTS.md, design docs, foundational cross-crate APIs
  require explicit ownership. Stop on dirty overlap or unnamed shared-file changes.
- **Handoff:** one outcome, exact paths, done criteria, constraints, focused checks,
  commit/push permission (default: neither). Report changed paths, results, risks,
  assumptions, unmade decisions.

## Project agents

Configured in `.pi/agents/<name>.md`. Default model: `deepseek/deepseek-v4-flash`.
`stage_owner` uses `deepseek/deepseek-v4-flash` (safe: delegates to Flash
worker children; orchestrator role alone does not require Pro).

| Agent | When to use | Model | Role |
| ------- | ------------- | ----- | ---- |
| `fast_explorer` | Read-only repo evidence & impact mapping | flash | scout |
| `spec_analyst` | Read-only design, pipeline, platform, benchmark, acceptance analysis | flash | analyst |
| `bounded_worker` | Exclusive-scope implementation + focused tests | flash | writer |
| `reviewer` | Read-only independent review of changes | flash | reviewer |
| `stage_owner` | Broader package, sole orchestrator, independent review, one verify pass | flash | orchestrator |

## Specialist skills

Defined in `.pi/skills/<name>/SKILL.md`. Loaded on demand by children; parent never reads full content.

| Skill | When to load |
| ------- | ------------- |
| `gate-verify` | Run `just check + test`, report exact failing gate/test. Read-only. |
| `ml-stage-debug` | Diagnose translation/OCR/inpaint Rust/Swift bridge failures |
| `subtitle-fetch-align-debug` | Diagnose provider, validation, ilass/xcorr alignment, dialogue classification |
| `benchmark-runner` | Audit stored ASR/translation benchmark regressions |

## Model cost policy

See `docs/ai/pi-harness-cost-policy.md` for full model ladder and cost discipline.

- Session lead: owner-selected model (expensive; judgment).
- Custom project agents: flash by default; oracle flash too; Pro explicit per-call hard reasoning only.
- Builtin reviewer, scout, worker, delegate, planner, context-builder, researcher: all flash (configured in `.pi/settings.json`).
