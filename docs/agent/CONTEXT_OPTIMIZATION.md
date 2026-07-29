# Context Optimization — Implementation Record

## Pre-Change Inventory

| Asset | Approximate scope |
| ------- | ------------------- |
| AGENTS.md | ~150 lines, project rules + agent catalogue |
| `.pi/agents/*.md` (4 agents) | ~50-100 lines each |
| `.pi/skills/` (5 skills) | 5 SKILL.md files, various sizes |
| `.pi/extensions/gate.ts` | ~160 lines TS |
| Active tools (stage_owner) | subagent, bash, read |
| Agent catalogue in AGENTS.md | 4 agents + 4 skills (1 deprecated unreferenced) |
| Child agent context | fresh, inheritProjectContext: true, inheritSkills: false |

## Baseline Token Measurements

Numeric measurements blocked: no token-counting tool available in this environment.
Pi version, model, provider, and context window not instrumented.
Estimate: fixed overhead ~2,000-3,000 tokens (AGENTS.md + agent defs + tool schemas).

Marked `blocked/unavailable` rather than inventing numbers.

## Target Architecture

```
Lean parent (orchestrator) → short rules, minimal tools, stable decisions
    ├── Child agents (execution) → role details, raw tool use, exploration
    │   └── Return → structured ≤1000-token handoff
    ├── Project memory files → decisions, state, plan, issues
    │   └── Survive compaction, parent reads relevant sections
    └── Tool output storage → full logs on disk, summaries in context
```

## Changes Implemented

1. `.pi/tool-output/` — ignored full-output storage, tracked README
2. `docs/agent/PROJECT_STATE.md` — seeded from AGENTS.md evidence
3. `docs/agent/DECISIONS.md` — 4 decisions with rationale
4. `docs/agent/CURRENT_PLAN.md` — context optimization tracking
5. `docs/agent/KNOWN_ISSUES.md` — 2 confirmed issues
6. `docs/agent/CONTEXT_OPTIMIZATION.md` — this file
7. `.pi/agents/*.md` — standardized handoff format (STATUS/SUMMARY/FILES/DECISIONS/VERIFICATION/RISKS/NEXT)
8. `AGENTS.md` — added output-hygiene + report rules, memory-file protocol
9. Removed `.pi/skills/pyo3-bridge-doctor/` — deprecated, zero references
10. Global Pi config: `contextPrune.enabled` true; backup at `~/.pi/agent/settings.json.bak-optimize`

## What Was NOT Changed

- `.pi/extensions/gate.ts` — outputs already compact (formatting errors only, not full logs). Left unchanged.
- No `disable-model-invocation` — unsupported.
- No `inheritProjectContext` change — it means AGENTS/context files, not parent transcript.
- No Context Inspector, Hypa, ReadSeek, Context Mode, Context Cap, or pi-agenticoding installed.
- No manifests/lockfiles touched.

## 3 Representative Workload Projections

| Workload | Expected parent calls | Expected child agents | Tool output strategy |
| ---------- | ---------------------- | ---------------------- | --------------------- |
| Small bug fix (1-2 files) | ~3-5 | 1 writer | Direct read, minimal output |
| Medium feature (several crates) | ~8-12 | 1 writer + 1 reviewer | Test logs to `.pi/tool-output/` |
| Repo-wide investigation | ~5-8 | 2-3 explorers | Grep results to file, summaries inline |

## Before/After Metrics

| Metric | Baseline | Post-Optimization | Notes |
| -------- | ---------- | ------------------- | ------- |
| Fixed prompt tokens | blocked | reduced (shorter agent defs, removed deprecated skill) | Cannot measure numerically |
| Agent handoff ceiling | unbounded | ≤1000 tokens | Enforced by format |
| Tool output in context | raw logs inline | summaries + file references | Convention established |
| Project memory surviving compaction | none | 4 curated files | Survives compaction as files |
| contextPrune.enabled | false | true | Existing config validated |

## Extension Decisions

| Extension | Decision | Reason |
| ----------- | ---------- | -------- |
| Context Inspector | Defer | No install capability in this session; blocked measurement anyway |
| Hypa | Defer | Evaluate after baseline measurement possible |
| ReadSeek | Defer | Whole-file reads not yet measured as top cost |
| Context Cap | Defer | After project memory + output hygiene proven |
| Context Mode | Defer | Overlaps with custom subagent framework |
| pi-agenticoding | Skip install, borrow patterns | Custom subagents already in place |

## Rollback

### Global config

```bash
cp ~/.pi/agent/settings.json.bak-optimize ~/.pi/agent/settings.json
```

### Repository files

```bash
git checkout -- AGENTS.md .gitignore .pi/agents/*.md docs/agent/
# Restore pyo3-bridge-doctor if needed:
git checkout -- .pi/skills/pyo3-bridge-doctor/
```

### No-op changes

- `.pi/tool-output/` is gitignored; delete directory if unwanted.
- `docs/agent/` can be removed: `rm -rf docs/agent/`

## Maintenance Guide

- **After major changes:** update `PROJECT_STATE.md` and `DECISIONS.md`.
- **When plan changes:** update `CURRENT_PLAN.md`.
- **When issues found/resolved:** update `KNOWN_ISSUES.md`.
- **After child agent runs:** enforce ≤1000-token handoff format.
- **Large tool output:** save to `.pi/tool-output/`, return summary.
- **Periodic audit:** check agent `.md` files haven't grown; trim if >1500 tokens.
- **New skills:** keep SKILL.md concise; prefer on-demand loading.
- **New agents:** use standardized handoff format. Keep parent-visible description to 1-2 sentences.
