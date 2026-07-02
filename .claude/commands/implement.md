---
description: Execute an implementation report produced by /plan, on Sonnet, end to end.
argument-hint: @<path-to-report>
model: claude-sonnet-4-6
allowed-tools: Read, Glob, Grep, Edit, Write, PowerShell, mcp__insitupy__*
---

# /implement — execute a plan report

Implement the change described in the report below. All conventions, environment rules, and
tool availability come from `CLAUDE.md` (auto-loaded into this session) and this command — the
report carries only the task.

## Report

$ARGUMENTS

## Tools available — `insitupy` MCP server

The `insitupy` MCP server exposes the source tree. **Tools are deferred: call `ToolSearch`
with `select:mcp__insitupy__<name>` to load a schema before first use.** Key tools:
`mcp__insitupy__get_function_source`, `mcp__insitupy__get_class_info`,
`mcp__insitupy__search_codebase`, `mcp__insitupy__get_data_model`. Prefer these over blind
file searching.

## Protocol

1. **Read the report fully** and confirm the goal, affected files, and acceptance criteria. If
   the report is missing, or contradicts the current code, stop and report that rather than
   guessing.
2. **Discover before writing.** Verify the report's file/line references against current source
   (they may have drifted since planning). Match surrounding conventions.
3. **Implement incrementally** — smallest coherent edits first.
4. **Run targeted tests** for the changed method/module only — not the full suite. Run
   `pytest <targeted test files>` from the environment where InSituPy is installed (see
   `CLAUDE.md`; the specific env name/path is in your own user config). Before *adding* a test,
   apply the altitude check: a test must exercise the **real code path against a real failure
   mode**. Do not add change-detector tests (asserting the specific line/attribute you wrote) or
   tests that stub the unit under test; for trivial state/glue changes, verify manually and note
   it in the log.
5. **Escalate when genuinely stuck:** for a hard design / security / correctness question, ask
   the user whether to invoke `@opus-consultant` (structured Opus Report contract). For a
   lightweight stall, the native `/advisor` toggle is the lower-friction option.
6. **Log the work** to `.log/log.md` (append via Edit, newest first) per the global CLAUDE.md
   format, and update `.log/backlog.md` if a tracked item was completed.

## Delegation note

Implement directly. Do not spawn subagents without the user's confirmation. If you need a
read-only lookup, the built-in `Explore` agent suffices — but restate any required convention
(the project environment, the targeted-test rule, the `insitupy` MCP tools) in the delegation
prompt, since subagents do **not** inherit `CLAUDE.md`.

Finish by reporting what changed, which tests ran, and their results.
