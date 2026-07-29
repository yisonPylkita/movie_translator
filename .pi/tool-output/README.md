# Tool Output Storage

Full command output saved here so model context stays compact.
Only summaries, errors, and file references enter conversation history.

## Naming pattern

```
.pi/tool-output/<category>-<YYYYMMDD-HHMMSS>.<extension>
```

Examples: `tests-20260728-161522.log`, `compiler-20260728-161744.log`.

## What goes here

| Source | Saved | Model-visible |
| -------- | ------- | --------------- |
| Test runs | Full log | Totals, failed names, key messages |
| Compiler/build | Complete output | Errors, warnings, exit code |
| Grep/ripgrep | Full results | 20-50 best matches |
| Git diff | N/A (Git retains) | Changed files + relevant hunks |
| MCP/API | Raw JSON | Selected fields, summary |

## Retention

- No automatic expiry. Delete when no longer needed for debugging.
- Gitignored except this README. Never commit logs.

## Recoverability

- Full output always on disk. Model can request specific ranges.
- Path references in conversation remain valid for the session.

## Secrets

- Never save tokens, keys, or passwords here.
- This directory is gitignored but not encrypted.
