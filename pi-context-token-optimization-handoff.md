# Pi / PyDev Context and Token Optimization Handoff

> **Purpose:** Paste the prompt in the first section into a fresh Pi session. The agent should inspect the current setup, determine which optimizations are actually applicable, and make a concrete implementation plan without blindly installing everything.

---

# 1. Paste-Ready Prompt for the Next Pi Session

```text
I want you to audit and optimize this Pi/PyDev agentic-coding environment for lower token usage and less context bloat.

Do not immediately install every extension mentioned below. First inspect the current environment, active tools, extensions, skills, AGENTS.md files, subagent definitions, MCP servers, context/compaction settings, and the way subagents are launched. Then identify the largest sources of repeated prompt tokens and tool-output tokens.

My goals, in priority order, are:

1. Reduce tokens repeatedly sent with every model request.
2. Reduce large tool outputs entering the conversation history.
3. Keep the main/orchestrator context small and stable.
4. Prevent child-agent exploration and logs from polluting the parent context.
5. Preserve enough project memory that compaction does not lose important decisions.
6. Avoid adding so many optimization tools that their own schemas and prompts create new bloat.
7. Keep the workflow convenient for normal coding work.

Please perform the work in the following phases.

PHASE 1 — MEASURE THE CURRENT STATE

Inspect and report:

- The approximate token contribution of the system prompt.
- Loaded AGENTS.md and other always-on instruction files.
- Skills advertised to the model.
- Active built-in tools.
- Extension-provided tools.
- MCP tool schemas and descriptions.
- Current conversation/session history.
- Typical output size from read, grep, shell, tests, compiler, git diff, browser, and MCP tools.
- How much parent context is copied into child agents.
- How much output child agents return to the parent.
- Current context-window and compaction configuration.
- Whether prompt caching is likely to be preserved or invalidated by dynamic tool changes.

If available and compatible, consider using Context Inspector:

    pi install npm:@vkundapur/context-inspector

After installation or if already present, inspect context with:

    /reload
    /context

Do not assume the extension is safe or compatible. Inspect its package/source/configuration first if practical, and tell me exactly what it adds to the prompt or tool registry.

At the end of Phase 1, produce a ranked table of the largest sources of token consumption. Separate:

- fixed tokens sent on nearly every request;
- accumulated conversation-history tokens;
- tool-result tokens;
- subagent input tokens;
- subagent output tokens;
- compaction and summary overhead.

PHASE 2 — MINIMIZE ALWAYS-ON TOOLS

Determine the smallest practical tool set for the parent/orchestrator.

The parent should probably have only tools similar to:

- read or targeted code retrieval;
- grep/search;
- bash or a constrained shell;
- subagent dispatch;
- context inspection;
- possibly a compact task-memory/notebook tool.

The parent should not automatically have every edit, deployment, browser, database, GitHub write, infrastructure, and MCP tool unless it needs them for the current task.

Investigate whether this Pi version supports:

    pi --tools read,grep,bash,subagent

or:

    pi --no-tools

and whether extensions can call something equivalent to:

    pi.setActiveTools([...])

Where practical, design role-specific tool sets:

ORCHESTRATOR
- read/search
- constrained shell
- subagent dispatch
- context information
- task memory

IMPLEMENTATION WORKER
- read/search
- edit/write
- shell
- tests

REVIEWER
- read/search
- shell
- no write tools by default

RESEARCHER
- search/retrieval tools only
- no repository mutation by default

DEPLOYMENT AGENT
- deployment tools only when explicitly invoked

Prefer deferred or dynamic tool loading where supported. A small discovery/router tool is preferable to exposing dozens or hundreds of schemas permanently.

Before changing anything, estimate the token savings from removing unused tool schemas.

PHASE 3 — REDUCE ALWAYS-ON AGENT DEFINITIONS

Inspect my subagent definitions and all AGENTS.md files.

For every instruction block, classify it as one of:

A. Must always be visible to the parent.
B. Only needed when selecting a specific agent.
C. Only needed inside that child agent.
D. Only needed for a particular task or skill.
E. Redundant, outdated, or duplicated.

Keep parent-visible agent descriptions extremely short: usually a name and one or two sentences explaining when to use the agent.

Move detailed methodologies, checklists, coding conventions, examples, test commands, and output rules into child-only prompts or on-demand skills.

Consider Pi skills under paths such as:

    .pi/skills/<skill-name>/SKILL.md

Use progressive disclosure. Only the short skill name and description should be always visible; detailed content should be loaded only when invoked.

For rarely used or very large skills, investigate whether this Pi version supports frontmatter similar to:

    ---
    name: rust-async-review
    description: Reviews Rust asynchronous code for cancellation, locking, lifetime, and task-management problems.
    disable-model-invocation: true
    ---

Such a skill should then be invoked explicitly, for example:

    /skill:rust-async-review

Do not move critical global safety or repository invariants out of always-on context. Distinguish genuinely global rules from role-specific detail.

PHASE 4 — CONTROL TOOL OUTPUT BEFORE IT ENTERS CONTEXT

The preferred strategy is to prevent bloat before compaction, not merely summarize it afterward.

Create or recommend compact result policies for each tool category.

Suggested normal limits:

- File read: relevant symbol, range, or 100–250 lines.
- Grep/search: 20–50 best matches with filenames and line numbers.
- Tests: totals, failed test names, essential failure messages, and the path to the full log.
- Compiler: errors and warnings only, deduplicated where possible.
- Git diff: changed-file summary plus only relevant hunks.
- Logs: 100–200 lines around errors, not the entire log.
- MCP/API: selected fields or a compact summary, not raw JSON by default.
- Browser/HTML: extracted relevant content rather than a full DOM snapshot.
- Subagent report: normally no more than about 1,000 tokens.

For large command output, save the complete output to a file and return only a compact model-visible result, for example:

    12 tests failed, 418 passed.

    Failures:
    - auth::tests::expired_token
    - auth::tests::concurrent_refresh

    Full output: .pi/tool-output/test-20260728-161522.log

The agent can inspect the saved log later if needed.

Check whether hidden shell execution is supported:

    !!some-command

Use it only for commands whose output I need to see or execute but the model does not need in context. Confirm the exact semantics in the installed Pi version before relying on it.

PHASE 5 — EVALUATE OUTPUT-REDUCTION EXTENSIONS

Evaluate, but do not automatically install, the following options.

OPTION A — Hypa

Possible installation:

    pi install npm:@hypabolic/pi-hypa

Potential benefits:

- deterministic reduction of shell output;
- compact test/build/compiler results;
- compact read, grep, find, and list operations;
- full output saved externally and recoverable;
- optional MCP proxy to avoid exposing all upstream schemas at once.

Prefer replace mode instead of duplicating built-in and Hypa tools, if compatible:

    {
      "mode": "replace",
      "mcpProxyEnabled": true
    }

Possible config location:

    ~/.hypa-pi/config.json

Possible launch override:

    HYPA_PI_MODE=replace pi

Before recommending it, verify:

- compatibility with this Pi version;
- what tools it registers;
- whether replace mode truly removes duplicate tools;
- whether it changes tool semantics important to my workflow;
- security implications;
- whether MCP proxying works with my actual servers;
- whether it preserves enough evidence for debugging.

OPTION B — ReadSeek

Possible installation:

    pi install npm:pi-readseek

Potential benefits:

- symbol-oriented reads;
- structural code maps;
- AST search;
- definition/reference navigation;
- constrained grep;
- anchored edits.

It may reduce whole-file reads, especially in large C++, Rust, Python, and TypeScript repositories.

However, check how many tools it registers. Prefer replacing overlapping built-ins or loading its tools dynamically. Do not keep many duplicate read/search/edit tools active without a measurable reason.

OPTION C — Context Mode

Possible installation:

    pi install npm:context-mode

Potential benefits:

- sandboxed processing of large results;
- local indexing and retrieval;
- SQLite/FTS-based session memory;
- retrieval of relevant fragments rather than raw history.

Treat vendor-reported token reductions as unverified until measured locally. Compare it against Hypa rather than installing both immediately.

OPTION D — Agenticoding-style clean contexts

Possible installation:

    pi install npm:pi-agenticoding

Potential benefits:

- clean child contexts;
- task notebooks;
- explicit handoffs;
- isolation of exploratory work.

My setup already has custom subagents, so first determine whether the architecture can be copied without installing the package.

At the end of this phase, provide a comparison table with:

- tokens saved;
- additional always-on schemas;
- operational complexity;
- debugging impact;
- security risk;
- overlap with existing tools;
- recommendation: install, test in isolation, borrow the design only, or skip.

PHASE 6 — IMPROVE SUBAGENT CONTEXT ISOLATION

Do not automatically copy the entire parent transcript into child agents.

A child should normally receive only:

- the concrete task;
- relevant file paths or symbols;
- applicable constraints;
- relevant architectural decisions;
- expected output format;
- required validation commands;
- a small context budget.

The child should read repository files directly when needed rather than receiving copies of large files in its prompt.

Design the parent/child flow like this:

PARENT CONTEXT
- current goal;
- accepted plan;
- stable decisions;
- concise progress state;
- concise worker reports.

CHILD CONTEXT
- detailed role instructions;
- task-specific source inspection;
- exploratory commands;
- raw logs;
- temporary hypotheses;
- implementation details.

Only distilled results should cross back to the parent.

Use the following default child-agent return contract unless a task explicitly needs something else:

    STATUS
    completed | blocked | needs-review

    SUMMARY
    Maximum 8 sentences.

    FILES
    - path: what changed

    DECISIONS
    - decision and brief rationale

    VERIFICATION
    - command
    - result

    RISKS
    - remaining uncertainty

    NEXT
    - maximum 5 concrete actions

Child agents must not return by default:

- full command output;
- full file contents;
- full diffs;
- their complete conversation;
- a narration of every step;
- repeated task descriptions;
- long generic explanations;
- speculative next steps unrelated to the assigned task.

Use a normal maximum of approximately 1,000 output tokens per child. Allow more only when the parent explicitly requests detailed evidence.

PHASE 7 — COMPACTION AND CONTEXT CAPS

Inspect the current model context size and Pi compaction configuration.

Compaction is a fallback mechanism. It should not replace good tool-output hygiene or child isolation.

Investigate current support and defaults for settings equivalent to:

    {
      "compaction": {
        "enabled": true,
        "reserveTokens": 24576,
        "keepRecentTokens": 12000
      }
    }

Verify the actual valid configuration keys before editing files.

For very large-context models, evaluate a lower operational cap so automatic compaction happens before the session grows to several hundred thousand tokens.

Possible extension:

    pi install npm:pi-context-cap

Possible policy:

    {
      "cap": 180000,
      "appliesOver": 200000,
      "matchPatterns": ["*"]
    }

Suggested working ranges to evaluate:

- 100k–150k: routine implementation and debugging;
- 150k–200k: broad repository work;
- above 200k: only when a specific task benefits enough to justify the cost.

Do not treat these numbers as universal. Measure model quality, cache behavior, summary quality, and actual API cost in my environment.

PHASE 8 — PERSIST STABLE MEMORY OUTSIDE CHAT HISTORY

Design a small project-memory system so compaction can discard raw exploration without losing important decisions.

Prefer a few maintained files over an ever-growing transcript, for example:

    docs/agent/PROJECT_STATE.md
    docs/agent/DECISIONS.md
    docs/agent/CURRENT_PLAN.md
    docs/agent/KNOWN_ISSUES.md

Suggested responsibilities:

PROJECT_STATE.md
- current architecture;
- major components;
- stable repository facts;
- important commands;
- current implementation status.

DECISIONS.md
- accepted decisions;
- rationale;
- rejected alternatives;
- date or commit when useful.

CURRENT_PLAN.md
- current objective;
- completed items;
- next items;
- blockers.

KNOWN_ISSUES.md
- confirmed defects;
- unresolved questions;
- reproduction steps;
- workarounds.

These files should remain concise and curated. They must not become raw transcript dumps.

Determine who updates them, at what points in the workflow, and how the parent agent reads only relevant sections.

PHASE 9 — IMPLEMENT INCREMENTALLY

Do not make many unrelated changes at once.

Recommended order:

1. Measure current token composition.
2. Remove unused always-on tools and duplicate schemas.
3. Shorten parent-visible subagent definitions.
4. Move detailed role instructions into child-only prompts or on-demand skills.
5. Introduce strict tool-output limits and full-log file storage.
6. Enforce concise child-agent reports.
7. Add project-memory files.
8. Test Hypa or another reducer in isolation.
9. Add ReadSeek only if whole-file inspection remains a major source of bloat.
10. Configure earlier compaction or a context cap.
11. Re-measure and compare against the baseline.

After each material change, collect:

- prompt/input tokens per representative turn;
- output tokens;
- tool-result sizes;
- number of active tools;
- tool-schema token estimate;
- cache hit/miss behavior if available;
- task completion quality;
- debugging convenience;
- wall-clock impact;
- cost per completed representative task.

Use at least three representative workloads:

A. Small bug fix in one or two files.
B. Medium feature spanning several components.
C. Repository-wide investigation or refactor planning.

PHASE 10 — DELIVERABLES

At the end, give me:

1. A baseline token-bloat report.
2. A ranked list of the largest causes.
3. A proposed target architecture.
4. Exact files and settings to change.
5. Minimal patches for those files.
6. Extensions recommended for installation, with reasons.
7. Extensions rejected or deferred, with reasons.
8. Before/after measurements where possible.
9. Risks and rollback instructions.
10. A concise maintenance guide for keeping the setup lean.

Important constraints:

- Do not install everything merely because it is mentioned.
- Inspect current versions and configuration syntax before changing files.
- Prefer removing or replacing tools over adding duplicate alternatives.
- Never discard full logs irreversibly; save them outside model context.
- Keep critical evidence recoverable.
- Do not weaken repository safety rules merely to save tokens.
- Do not make destructive configuration changes without a backup or patch.
- Show me the proposed modifications before applying changes that materially alter the workflow.
- Optimize cost per completed task, not only tokens per individual request.
```

---

# 2. Recommended Target Architecture

The desired architecture is not simply “use aggressive compaction.” It is a layered system that prevents irrelevant information from entering the expensive context in the first place.

```text
                        ┌──────────────────────────┐
                        │       User Request       │
                        └────────────┬─────────────┘
                                     │
                                     ▼
                        ┌──────────────────────────┐
                        │   Lean Parent Agent      │
                        │                          │
                        │ - short global rules     │
                        │ - minimal active tools   │
                        │ - stable decisions       │
                        │ - current plan           │
                        └───────┬──────────┬───────┘
                                │          │
              task-specific     │          │ targeted retrieval
              context only      │          │
                                ▼          ▼
                  ┌──────────────────┐  ┌──────────────────┐
                  │   Child Agent    │  │ Project Memory   │
                  │                  │  │                  │
                  │ - role details   │  │ decisions        │
                  │ - raw tool use   │  │ current state    │
                  │ - exploration    │  │ known issues     │
                  │ - implementation │  │ current plan     │
                  └────────┬─────────┘  └──────────────────┘
                           │
                           │ concise structured report
                           ▼
                  ┌──────────────────┐
                  │  Parent Context  │
                  │  remains small   │
                  └──────────────────┘

Large tool output is written to files, indexes, or logs outside the chat.
Only summaries, errors, relevant snippets, and file references enter model context.
```

---

# 3. Core Principles

## 3.1 Prevent bloat before compaction

A 20,000-token compiler or test output can be resent on many later turns. Summarizing it after ten turns does not recover the tokens already spent. The best intervention point is before the raw output enters the conversation.

## 3.2 Minimize fixed per-request overhead

Anything always present is multiplied by the number of model calls:

- system instructions;
- AGENTS.md files;
- skill descriptions;
- agent catalogues;
- tool descriptions;
- JSON schemas;
- MCP catalogues.

Removing 5,000 fixed tokens from a 100-call session can matter more than shortening a single response by 20,000 tokens.

## 3.3 Separate coordination from execution

The parent agent should coordinate. Child agents should perform noisy investigation and implementation. Raw exploration should not return to the parent unless it is essential evidence.

## 3.4 Keep full evidence recoverable

Token optimization must not make debugging impossible. Save full logs and raw outputs to files, then give the model concise summaries plus paths to the original evidence.

## 3.5 Measure completed-task cost

A model using fewer tokens per turn may take more turns, make more mistakes, or force repeated file reads. Compare total cost and quality for completed tasks rather than optimizing a single request in isolation.

---

# 4. Suggested Tool-Output Policy

| Tool or result | Default model-visible output | Full result handling |
|---|---|---|
| File read | Requested symbol or 100–250 relevant lines | Read additional ranges on demand |
| Grep/search | 20–50 ranked matches | Save full results when unusually large |
| Test suite | Passed/failed totals, failed names, key messages | Save full log under `.pi/tool-output/` |
| Compiler/build | Errors, warnings, exit code, deduplicated causes | Save complete build log |
| `git diff` | Changed files and relevant hunks | Full diff remains available through Git |
| Runtime logs | Error neighborhood and recent tail | Save or retain original log file |
| MCP/API | Selected fields and concise summary | Save raw JSON if needed |
| Browser page | Relevant extracted text | Save snapshot only when useful |
| Subagent | Structured report, normally ≤1,000 tokens | Child session/log retained separately |

A suitable naming pattern is:

```text
.pi/tool-output/<category>-<YYYYMMDD-HHMMSS>.<extension>
```

Examples:

```text
.pi/tool-output/tests-20260728-161522.log
.pi/tool-output/compiler-20260728-161744.log
.pi/tool-output/mcp-github-20260728-162101.json
```

---

# 5. Reusable Parent-to-Child Prompt Template

```text
ROLE
You are the <ROLE NAME> for this task.

TASK
<Concrete task with one clear outcome.>

RELEVANT CONTEXT
- <Only the architectural decisions needed for this task.>
- <Relevant file paths, symbols, or components.>
- <Known constraints.>

DO NOT ASSUME
- Do not assume the rest of the parent conversation is relevant.
- Inspect repository files directly when additional context is needed.
- Do not broaden the task without reporting a blocker.

TOOLS
Use only the tools required for this role. Do not invoke unrelated tools.

OUTPUT BUDGET
Keep the final report under approximately 1,000 tokens unless detailed evidence is explicitly requested.

DO NOT RETURN
- full command output;
- complete file contents;
- full diffs;
- narration of every step;
- repeated task description;
- generic background explanations.

RETURN EXACTLY THIS STRUCTURE

STATUS
completed | blocked | needs-review

SUMMARY
Maximum 8 sentences.

FILES
- path: what changed or what was inspected

DECISIONS
- decision and short rationale

VERIFICATION
- command: result

RISKS
- remaining uncertainty

NEXT
- maximum 5 concrete actions
```

---

# 6. Reusable Implementation-Agent Prompt

```text
You are an implementation worker operating in an isolated child context.

Implement only the assigned change. Read the relevant files directly. Do not request or reproduce the full parent conversation.

Before editing:

1. Identify the smallest set of files required.
2. Confirm existing local conventions from nearby code.
3. State any blocking ambiguity briefly; otherwise make a reasonable repository-consistent choice.

During implementation:

- Prefer minimal, reviewable changes.
- Do not reformat unrelated code.
- Do not modify generated or vendored files unless required.
- Run the narrowest useful validation first.
- Store large compiler and test output in a log file.
- Return only failure summaries and relevant diagnostics to the parent.

Final response limit: approximately 1,000 tokens.

Use this exact report format:

STATUS
completed | blocked | needs-review

SUMMARY
Maximum 8 sentences.

FILES
- path: change

DECISIONS
- decision and rationale

VERIFICATION
- command: result

RISKS
- remaining uncertainty

NEXT
- maximum 5 actions
```

---

# 7. Reusable Reviewer Prompt

```text
You are a read-only code reviewer in an isolated child context.

Review only the assigned change. Do not rewrite the implementation unless explicitly asked. Inspect the changed files and the minimum surrounding code needed to verify behavior.

Prioritize:

1. Correctness defects.
2. Data loss, races, deadlocks, security problems, and unsafe behavior.
3. Broken invariants and incompatible interfaces.
4. Missing error handling.
5. Insufficient tests for realistic failure cases.
6. Material performance regressions.

Do not spend report space on minor style preferences unless they obscure a real defect.

For every finding include:

- severity;
- file and line or symbol;
- concrete failure scenario;
- recommended correction.

Do not include full files, full diffs, or a narration of the review process.

Keep the final response under approximately 1,000 tokens unless severe findings require more evidence.

Return:

STATUS
completed | blocked | needs-review

SUMMARY
Maximum 8 sentences.

FINDINGS
- severity — path:symbol — problem — failure scenario — correction

VERIFICATION
- command or inspection: result

RISKS
- remaining uncertainty

NEXT
- maximum 5 actions
```

---

# 8. Reusable Research-Agent Prompt

```text
You are a repository research agent in an isolated child context.

Answer the assigned technical question by inspecting only relevant code, documentation, configuration, history, and tests. Prefer targeted search, symbol reads, and structural navigation over reading entire files.

Do not modify the repository.

When command or search output is large:

- save the complete result externally;
- return only relevant matches and a path to the full output;
- avoid pasting raw JSON, logs, or entire files.

Distinguish:

- confirmed facts;
- strong inferences;
- unresolved questions.

Keep the final response under approximately 1,000 tokens.

Return:

STATUS
completed | blocked | needs-review

ANSWER
Maximum 8 sentences.

EVIDENCE
- path:symbol or command: concise evidence

INFERENCES
- inference and supporting facts

UNRESOLVED
- remaining uncertainty

NEXT
- maximum 5 actions
```

---

# 9. Suggested Project-Memory Files

## `docs/agent/PROJECT_STATE.md`

```markdown
# Project State

## Architecture
- Concise description of major components.

## Stable Invariants
- Rules that must remain true.

## Important Commands
- Build:
- Test:
- Lint:
- Run:

## Current Implementation Status
- Component: status.

## Repository Conventions
- Only conventions that repeatedly affect agent work.
```

## `docs/agent/DECISIONS.md`

```markdown
# Decisions

## YYYY-MM-DD — Decision title

**Decision:**
Concise accepted choice.

**Reason:**
Why it was selected.

**Alternatives rejected:**
- Alternative: short reason.

**Consequences:**
- Important follow-up or constraint.
```

## `docs/agent/CURRENT_PLAN.md`

```markdown
# Current Plan

## Objective
One concise objective.

## Completed
- Item.

## In Progress
- Item.

## Next
- Item.

## Blockers
- Blocker or `None`.
```

## `docs/agent/KNOWN_ISSUES.md`

```markdown
# Known Issues

## Issue title

**Status:** confirmed | suspected | resolved

**Symptoms:**
- Concise symptom.

**Reproduction:**
1. Step.

**Evidence:**
- Path, test, or log reference.

**Workaround:**
- Temporary workaround or `None`.

**Next action:**
- Concrete action.
```

---

# 10. Extension Evaluation Matrix

| Option | Primary benefit | Main risk | Initial recommendation |
|---|---|---|---|
| Context Inspector | Shows what consumes context | Adds its own extension code and possibly tools | Install or test first; measure before optimizing |
| Hypa | Reduces shell/tool/MCP output before context insertion | Changes tool behavior; possible duplication if not in replace mode | Strong candidate; test in isolation |
| ReadSeek | Targeted symbol and AST navigation | May register many overlapping tools | Add only if whole-file reads remain a major problem |
| Context Cap | Forces earlier compaction on very large windows | Poor summaries may discard useful context too early | Use after project memory and output hygiene are in place |
| Context Mode | Indexes and retrieves context externally | More opinionated and operationally complex | Compare against Hypa; do not install both initially |
| pi-agenticoding | Clean child contexts and handoffs | May overlap with custom subagents | Borrow architecture first; install only for missing features |

---

# 11. Baseline and Benchmark Worksheet

Run the same representative task before and after changes.

```markdown
# Context Optimization Benchmark

## Environment
- Pi version:
- Model:
- Provider:
- Context window:
- Active extensions:
- Active MCP servers:

## Workload
- Small bug fix:
- Medium feature:
- Repository-wide investigation:

## Measurements

| Metric | Baseline | Optimized | Difference |
|---|---:|---:|---:|
| Initial fixed prompt tokens |  |  |  |
| Active tool-schema tokens |  |  |  |
| AGENTS/skills tokens |  |  |  |
| Average input tokens per model call |  |  |  |
| Total input tokens per completed task |  |  |  |
| Total output tokens per completed task |  |  |  |
| Largest tool result |  |  |  |
| Number of model calls |  |  |  |
| Number of child agents |  |  |  |
| Parent-to-child input tokens |  |  |  |
| Child-to-parent output tokens |  |  |  |
| Compactions |  |  |  |
| Cache hit rate, if available |  |  |  |
| Task outcome quality |  |  |  |
| Manual debugging convenience |  |  |  |
| Total API cost |  |  |  |
```

---

# 12. Practical Priority Order

1. Measure context composition.
2. Remove unused and duplicated tool schemas.
3. Shorten the parent-visible agent catalogue.
4. Move detailed instructions into child-only prompts or explicit skills.
5. Limit tool outputs and store full logs outside context.
6. Enforce structured subagent reports with a normal 1,000-token ceiling.
7. Introduce concise project-memory files.
8. Test Hypa in replace mode.
9. Add structural code retrieval only if measurements justify it.
10. Configure earlier compaction or a lower operational context cap.
11. Re-run the benchmark and compare cost per completed task.

---

# 13. Likely High-Impact Starting Configuration

This is a hypothesis to test, not a command to apply blindly:

```text
Lean orchestrator tool set
+
short parent-visible agent definitions
+
child-only detailed prompts
+
full logs written outside context
+
compact tool-result summaries
+
1,000-token child handoffs
+
curated project memory
+
operational context cap around 150k–200k
```

A plausible first extension combination is:

```text
Context Inspector
+
Hypa in replace mode
+
Context Cap after baseline measurement
```

ReadSeek should be added only when measurements show that broad file reads and textual search remain a major cost source.

Context Mode and pi-agenticoding should first be evaluated as architectural patterns, because they may overlap with an existing custom subagent framework.

---

# 14. Anti-Patterns to Avoid

- Installing many context extensions simultaneously.
- Keeping built-in and replacement tools active together without a reason.
- Advertising every MCP tool schema to every agent.
- Copying the entire parent transcript into each child.
- Returning complete child transcripts to the parent.
- Pasting full compiler, test, browser, or API output into the conversation.
- Reading complete files when a symbol or range is sufficient.
- Treating a million-token context window as a target size.
- Relying on compaction while leaving tool-output bloat unchanged.
- Creating project-memory files that become uncurated transcript dumps.
- Optimizing input tokens while increasing retries and reducing task quality.
- Irreversibly truncating evidence needed for debugging.

---

# 15. Definition of Success

The optimization is successful when:

- the parent’s fixed prompt and tool-schema cost is materially lower;
- common commands return compact results;
- full logs remain recoverable outside context;
- child agents receive only task-specific context;
- child reports are concise and structured;
- stable decisions survive compaction through curated files;
- representative tasks cost fewer total tokens or less money to complete;
- implementation quality and debugging convenience do not materially degrade;
- the setup remains understandable and maintainable.
