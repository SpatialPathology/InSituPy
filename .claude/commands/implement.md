---
description: Execute an implementation report produced by /plan, /plan-opus, or /plan-fable, on Sonnet, end to end.
argument-hint: @<path-to-report>
model: claude-sonnet-5-5
allowed-tools: Read, Glob, Grep, Edit, Write, PowerShell, mcp__insitupy__*
---

# /implement — execute a plan report

@.claude/implement-protocol.md

## Report

$ARGUMENTS

## Tools available — `insitupy` MCP server

The `insitupy` MCP server exposes the source tree. **Tools are deferred: call `ToolSearch`
with `select:mcp__insitupy__<name>` to load a schema before first use.** Key tools:
`mcp__insitupy__get_function_source`, `mcp__insitupy__get_class_info`,
`mcp__insitupy__search_codebase`, `mcp__insitupy__get_data_model`. Prefer these over blind
file searching.

## Your role: interactive session

1. **Agent plans, at the start** (report under `agent-plans/`, written by the scheduled plan
   agent): if the report is still in a date folder (`agent-plans/<YYMMDD>/...`), it has not been
   reviewed. Say so and ask whether to proceed or to review it first with
   `/walkthrough-insitupy-plans`.
2. **Escalate when genuinely stuck:** for a hard design / security / correctness question, ask
   the user whether to invoke a senior-advisor agent if one is configured (e.g. the maintainer's
   `@opus-consultant`, a structured Opus Report contract); otherwise surface the question to a
   human. For a lightweight stall, the native `/advisor` toggle is the lower-friction option.
3. **Log the work.** Check whether `.log/log.md` and `planning/backlog.md` already exist at the
   repo root.
   - If they exist, append an entry to `.log/log.md` (via Edit, newest first) per the global
     CLAUDE.md format, and update `planning/backlog.md` if a tracked item was completed.
   - If they don't exist, don't create them silently. Ask the user whether to set up this
     session-log/backlog structure (briefly: a running dev journal at `.log/log.md` and a todo
     list at `planning/backlog.md`); if they decline, skip logging entirely.
4. **Agent plans, at the end:** once the change is implemented and the targeted tests pass, move
   the plan folder from `agent-plans/approved/<YYMMDD>-<slug>/` to
   `agent-plans/implemented/<YYMMDD>-<slug>/`, and update the plan's row in
   `agent-plans/review-log.md` (outcome `implemented`, date, branch, new location). If parts
   were left out, keep the folder in `approved/`, note what is missing in the row, and ask
   the user.

## Delegation note

Implement directly. Do not spawn subagents without the user's confirmation. If you need a
read-only lookup, the built-in `Explore` agent suffices — but restate any required convention
(the project environment, the targeted-test rule, the `insitupy` MCP tools) in the delegation
prompt, since subagents do **not** inherit `CLAUDE.md`. To implement several approved agent plans
at once, use `/implement-insitupy-plans` instead.

Finish by reporting what changed, which tests ran, and their results.
