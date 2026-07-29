# Current Plan

> Owner: parent orchestrator. Update when objective changes or items complete.

## Objective

Context/token optimization — reduce per-request overhead and tool-output bloat.

## Completed

- Context optimization package: standardized agent handoffs, project memory files, tool-output storage convention, global prune config enabled.
- Removed deprecated `pyo3-bridge-doctor` skill.

## In Progress

- None.

## Next

- Measure token composition after optimization.
- Evaluate Hypa in replace mode if shell/test output still bloated.
- Consider ReadSeek if whole-file reads remain major cost.
- Benchmark 3 representative workloads (small fix, medium feature, repo-wide investigation).

## Blockers

- None.
