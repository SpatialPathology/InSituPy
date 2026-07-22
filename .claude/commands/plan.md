---
description: Plan a change with Opus (read-only) and write a self-contained implementation report to .log/.
argument-hint: <goal or task description>
model: opus
allowed-tools: Read, Glob, Grep, Write, WebSearch, WebFetch, PowerShell, mcp__insitupy__*
---

# /plan — design an implementation, write a durable report

You are planning the task below. **Make no code edits.** Your only deliverable is a
self-contained report written to `.log/`.

## Task

$ARGUMENTS

## Protocol

1. **Understand the codebase before planning.** Use the `insitupy` MCP server
   (`mcp__insitupy__get_data_model`, `get_function_source`, `search_codebase`, … — load
   schemas via `ToolSearch` `select:` first) and read the relevant source. Prefer MCP
   introspection over blind file searching.
2. **Design the change.** Identify the files to touch, the approach, edge cases, and the
   targeted tests that will need to run (per project convention; tests run from the
   environment where InSituPy is installed).
3. **Surface hard design / architecture / security questions** to the user and ask whether to
   consult a senior-advisor agent if one is configured (e.g. the maintainer's
   `@opus-consultant`); otherwise surface the question to a human. Do not escalate automatically.
4. **Write the report.** Check whether `.log/` already exists at the repo root.
   - If it exists, write to `.log/reports/YYMMDD/<short-task-title>/report-<short-task-title>.md`
     (date prefix via `printf '%(%y%m%d)T\n' -1`, or read `currentDate` from context). Use the
     Write tool to create subdirectories as needed.
   - If it does **not** exist, don't create it silently. Ask the user whether to set it up,
     showing the path it would create (`.log/reports/YYMMDD/<short-task-title>/report-....md`);
     if they decline, write the report to a location they specify instead (or keep it in the
     conversation if there is none).

## Report contents (must be self-contained)

The report is the handoff to `/implement`, possibly in a different session days later. It must
stand alone without this conversation. Include:

- **Goal** — what and why, in plain language.
- **Affected files** — each with the specific change.
- **Approach** — ordered, step by step.
- **Edge cases & risks.**
- **Tests** — which targeted tests to run or add, **and at what altitude**: for each change,
  say whether it warrants a test (real failure-mode logic) or manual verification (trivial
  state/glue). Don't prescribe change-detector or stub-the-unit tests. Run with `pytest` from
  the project env.
- **Acceptance criteria.**
- **Open questions / decisions** — with any resolutions already made.

Do **not** include behavioral boilerplate (conventions, MCP tool inventory, environment rules)
— that lives in `CLAUDE.md` and the `/implement` command. The report carries only the task.

End by printing the report path so it can be passed to `/implement @<path>`.
