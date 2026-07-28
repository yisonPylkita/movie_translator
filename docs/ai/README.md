# AI assistance assets

Live project AI configuration adapted from sibling `energy-orchestrator` repository.

## Live surfaces

| Path | Purpose |
| --- | --- |
| `AGENTS.md` | Repository facts, invariants, commands, and agent policy |
| `.pi/settings.json` | Pi subagent model ladder |
| `.pi/agents/*.md` | Pi project agents |
| `.pi/skills/*/SKILL.md` | Domain-specific diagnostics and verification skills |
| `.pi/extensions/*.ts` | Pi gate/checkpoint/todo integration |
| `.pi/rules/**` | Rust import and style checks |
| `docs/ai/agent-orchestration.md` | Work-package, prompt, review, and handoff protocol |
| `docs/ai/session-prompt.md` | Pasteable fresh-session prompt template |
| `docs/ai/pi-harness-cost-policy.md` | Pi model/cost routing policy |

## Source adaptation

Copied concepts:

- bounded exclusive-scope writers;
- read-only repository and specification analysts;
- one-stage owner orchestration;
- task envelopes, structured handoffs, independent review, and blocker escalation;
- cheap-worker model pins and expensive-lead cost discipline.

Replaced energy-specific assumptions with movie-translator constraints:

- pure Rust workspace; no Python/PyO3;
- one serialized GPU worker;
- macOS Apple Translation/Vision/Speech bridges;
- provider rate limits and external/browser boundaries;
- multiple viable subtitle candidates;
- `just check && just test` full gate.

Not copied:

- `.pi-subagents/artifacts/`: generated transcripts/results, not configuration;
- HEO sprint plans, milestone prompts, ADR/requirement IDs, physical-control safety text, or current-state ledgers: project-specific and misleading here;
- stale historical prompts: reusable mechanics live in current docs instead.

Update live docs/config when repository architecture or commands change. Do not create session-state prompt files containing guessed branch state or fake completion claims.
