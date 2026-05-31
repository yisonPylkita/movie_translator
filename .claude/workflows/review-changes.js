// review-changes — read-only, parallel, lane-partitioned diff review.
//
// Mirrors the manual 3-lane review this repo standardized on: the codebase
// splits into three DISJOINT file lanes that never share files —
//   1. Rust crates       (crates/**, Cargo.toml, Cargo.lock)
//   2. Python ML backend (movie_translator/**, conftest.py)
//   3. tooling / docs     (justfile, .github/**, docs/**, scripts/**, pyproject.toml)
// so a reviewer can read each lane in isolation without cross-talk. We fan out
// one agent per lane over the working-tree diff, then synthesize a single
// prioritized report.
//
// Strictly read-only: every agent reviews via `git diff` / `git diff
// --staged` and reports findings. NONE of them edit, stage, or commit.

export const meta = {
  name: 'review-changes',
  description:
    'Read-only 3-lane parallel review of the working-tree diff (Rust / Python ML / tooling), then a prioritized synthesis.',
  phases: [{ title: 'Review' }, { title: 'Synthesize' }],
};

const READONLY = [
  'STRICTLY READ-ONLY: inspect with `git diff` and `git diff --staged` only.',
  'Do NOT edit, write, stage, commit, or push anything. If there is no diff in',
  'your lane, say so and return an empty findings list.',
].join(' ');

const lane = (name, globs, extra) =>
  [
    'You are the ' + name + ' reviewer for this repo.',
    'Your lane is ONLY these files: ' + globs + '.',
    'Run `git diff -- ' + globs + '` and `git diff --staged -- ' + globs + '`',
    'to see the changed files in your lane, and review ONLY those.',
    'Ignore changes outside your lane (another agent owns them).',
    extra,
    'Hunt for correctness bugs and security issues. For each finding return:',
    'file:line, severity (high/med/low), what is wrong, and the concrete fix.',
    READONLY,
  ].join(' ');

phase('Review');
const findings = await parallel([
  () =>
    agent(
      lane(
        'rust-crates',
        'crates/** Cargo.toml Cargo.lock',
        'Watch for: clippy-level correctness issues (unwrap/expect on fallible ' +
          'paths, panics in library code), broken GPU serialization (calling ' +
          '`mt_ml::{translate,ocr_*,inpaint}` off the single worker or off a ' +
          'runtime thread instead of spawn_blocking), holding the mt-ml modules ' +
          'mutex across a re-entrant Python call (deadlock), changed PyO3 ' +
          'embedding/init invariants, and error-handling that swallows the ' +
          'Python stderr-log path.',
      ),
      { label: 'rust', phase: 'Review' },
    ),
  () =>
    agent(
      lane(
        'python-ml',
        'movie_translator/** conftest.py',
        'Watch for: ruff-level issues, ML-stage logic bugs (translation merger ' +
          'timing, OCR change-detection/dedup, inpaint mask), platform guards ' +
          '(Apple Vision is macOS-only — Linux must degrade, not crash), ' +
          'model-load caching regressions, and anything that breaks the ' +
          '`import movie_translator` contract the Rust bridge depends on.',
      ),
      { label: 'python', phase: 'Review' },
    ),
  () =>
    agent(
      lane(
        'tooling-docs',
        'justfile .github/** docs/** scripts/** pyproject.toml rust-toolchain.toml',
        'Watch for: CI that re-enables the git-lfs model fetch (deliberately ' +
          'off), a build order that runs cargo before `uv sync` (PyO3 needs the ' +
          'venv first), a dropped `PYO3_PYTHON`/`LD_LIBRARY_PATH` setup, ' +
          'just recipes that lost a gate, shell scripts with unquoted ' +
          'expansions, and docs that contradict the code.',
      ),
      { label: 'tooling', phase: 'Review' },
    ),
]);

phase('Synthesize');
const report = await agent(
  [
    'Merge these three lane reviews into ONE prioritized report.',
    'Order strictly by severity (high -> med -> low); within a severity,',
    'group by lane. Dedupe overlapping findings. For each: file:line,',
    'severity, problem, fix. End with a one-line verdict: SHIP / FIX-FIRST /',
    'BLOCKED. Do not edit any files.',
    'Lane findings JSON: ' + JSON.stringify(findings),
  ].join(' '),
  { label: 'synth', phase: 'Synthesize' },
);

return report;
