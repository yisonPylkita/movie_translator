// benchmark-audit — read-only audit of the stored translation-quality
// benchmark results in git.
//
// Mirrors the `benchmark-runner` subagent's DEFAULT (read-only) mode: read the
// result files under benchmarks/ and their git history, report current scores
// and the delta vs the last committed run, and flag regressions with the
// commit that introduced them. STRICTLY read-only — it does NOT execute a real
// (GPU-bound) benchmark run, install anything, or commit. Running the benchmark
// is an explicit, slow, GPU-bound action; this workflow only audits what's
// already stored.

export const meta = {
  name: 'benchmark-audit',
  description:
    'Read-only audit of stored benchmark results in git: current scores, delta vs last run, flag regressions.',
  phases: [{ title: 'Audit' }],
};

phase('Audit');
const summary = await agent(
  [
    'Audit the translation-quality benchmark results stored in this repo.',
    'STRICTLY READ-ONLY — do not run the benchmark, do not `uv sync`, do not',
    'edit or commit anything.',
    '',
    'Step 1 — locate results: `ls -R benchmarks/` and find the stored score',
    'files (JSON/CSV/markdown — read whatever format is actually there; do not',
    'assume). Read the benchmark scripts (e.g. benchmarks/onepiece/) only to',
    'understand what each metric means.',
    '',
    'Step 2 — history: `git log --oneline -- benchmarks/` to find when results',
    'last changed; use `git show <rev>:<path>` to read previous score files and',
    'compute the delta vs the current committed values.',
    '',
    'Step 3 — flag regressions: any metric that dropped run-over-run. If',
    'discoverable, name the commit that introduced the drop. Also note stale',
    'results (benchmarks/ unchanged across many translation-stage commits) — a',
    'sign the benchmark was not re-run after a refactor, contrary to repo',
    'convention.',
    '',
    'Return a compact summary: current scores per metric/dataset, delta vs the',
    'previous committed run (▲/▼), a Regressions list, and any staleness note.',
    'Do NOT modify anything.',
  ].join('\n'),
  { label: 'audit', phase: 'Audit' },
);

return summary;
