---
name: walkthrough-insitupy-plans
description: Review the scheduled plan agent's pending plans in agent-plans/ with the maintainer, one plan at a time - approve, defer or reject each, answer its open questions, then file the outcome (report, status folder, review log, backlog). Use on /walkthrough-insitupy-plans, "review the agent plans" or "walk me through the agent plans".
---

# Walkthrough of the scheduled agent's plans

The scheduled plan-only agent (see "Unattended runs" in `CLAUDE.md`) writes plans to
`agent-plans/`. This skill turns each pending plan into a decision by the maintainer and files the
result where both the maintainer and the next agent run will find it. Claude explains and
recommends; the maintainer decides.

## Layout of `agent-plans/`

`agent-plans/` is a local link to a private cloud folder (see `.claude/setup-notes.md`). If it is
missing, stop and say so.

| Path | Meaning |
|---|---|
| `<YYMMDD>/` | **Inbox.** One folder per agent run: `run-summary.md` plus the plans not reviewed yet (`<slug>/report-<slug>.md`). The agent only ever writes here. |
| `approved/<YYMMDD>-<slug>/` | Approved, waiting for `/implement` (also "go with changes"). |
| `deferred/<YYMMDD>-<slug>/` | Good idea, wrong time. The review log names what unblocks it. |
| `rejected/<YYMMDD>-<slug>/` | Not to be done. Kept so the reasoning is not lost. |
| `implemented/<YYMMDD>-<slug>/` | Implemented on a branch. Moved here by `/implement`, not by this skill. |
| `review-log.md` | One row per reviewed plan: outcome, reason or unblock condition, location. **The feedback channel to the agent** - it reads this file before picking work. |
| `format-watch.md` | Agent state for the 10x format watch. Do not edit here unless the maintainer asks. |

## Step 0: Load the walkthrough style

Invoke the `anthropic-skills:walkthrough` skill (Skill tool) and follow its per-decision format
and style for every decision below. If it is unavailable, use this fallback: one decision at a
time; for each, what is decided and why it matters, the background needed, 2-4 options with
trade-offs, reversibility, a recommendation; then ask with AskUserQuestion, recommended option
first. This skill adds the scope, the decisions and the filing rules.

## Step 1: Collect

1. Find pending items: unticked review lines (`- [ ] ... #agent-review`) in the date folders
   (`agent-plans/[0-9]*/`). Two kinds:
   - plan reports (`<slug>/report-<slug>.md`, line 3);
   - a manual-check reminder in a `run-summary.md` (format watch). Treat it as a reminder, not a
     plan: mention it in the overview, tick it only when the maintainer says it is done.
2. Read each `run-summary.md` of the runs involved, then each pending report fully (`/implement`
   will rely on it, so read it the way the implementer will). Note: verdict, scope, open
   questions, the agent's recommendations, and dependencies (branches, other plans, releases).
3. Check what changed since the plan was written - plans go stale. For example: is a branch it
   depends on merged now (`git log`, `git branch -a`), does the backlog item still exist, did
   another plan or commit already cover part of it. Report drift in the overview.
4. Also read `review-log.md`. For `deferred` rows whose unblock condition now looks met, offer
   them for re-review at the end.
5. Show the overview: a numbered table (plan, run date, agent verdict, open questions, due date,
   dependencies or drift), ordered so plans others depend on come first. State the count, then
   start with #1. The maintainer may pick a subset; unpicked plans stay in the inbox untouched.

## Step 2: One plan at a time

For each plan:

1. **Outcome.** Summarize the plan in 3-5 lines (what, why, the agent's verdict and its key
   argument, effort, risk). Then ask: **approve** / **approve with changes** / **defer** /
   **reject** / **skip for now** (stays in the inbox). Recommend based on the report and the
   drift found in step 1, not just the agent's verdict.
2. **If approved:** walk through the open questions one by one (agent's recommendation as the
   default option unless drift argues otherwise). Then ask about changes to scope if the
   maintainer hinted at any, and about PR shape if the report leaves it open (separate PRs,
   target branch `main` vs a backport to `release/0.12.x`).
3. **If deferred:** ask for the unblock condition (concrete: "after `feat/read-atera` merges",
   "after the 2026-11-27 release", "when 10x ships X"). Open questions stay unanswered.
4. **If rejected:** get a one-line reason, and ask whether the underlying backlog item goes too
   (**drop the item**) or stays for later human work (**keep, not for the agent**).
5. Partial outcomes are fine ("Part A approved, Part B deferred"). File the plan as approved and
   record the deferred part in the review section and as its own backlog line.

## Step 3: File the outcome (right after each plan, not batched at the end)

Filing per plan keeps the progress if the session stops halfway. Use Edit for edits, PowerShell
for moves. Do not reformat other content in these files.

1. **Report.**
   - Tick the review line: `- [ ]` -> `- [x]`, and append ` ✅ YYYY-MM-DD` at the very end of
     the line (Obsidian Tasks done date).
   - Insert a section directly below the review line:
     ```
     ## Review (YYYY-MM-DD)

     **Outcome:** approved | approved with changes | deferred | rejected - one-line reason.
     This section overrides the rest of the report where they conflict.

     - Open question 1: <answer>
     - Open question 2: <answer>
     - Changes to scope: <...>        (only if any)
     - Deferred parts: <...>          (only if any)
     - Unblock condition: <...>       (deferred only)
     ```
   - Resolve any `JW:` annotations already in the report into this section (global convention).
2. **Move** the plan folder (not the run summary):
   `Move-Item agent-plans/<YYMMDD>/<slug> agent-plans/<status>/<YYMMDD>-<slug>`.
   Create the status folder if missing. Leave the date folder in place even when only
   `run-summary.md` remains - it is the run record.
3. **Review log.** Add a row at the top of the table in `agent-plans/review-log.md` (newest
   first): date, `<YYMMDD>-<slug>`, outcome, reason (rejected), unblock condition (deferred) or
   short note (approved), link to the new location. Write the reason so the agent can act on it:
   "do not re-plan: X" or "re-plan when Y".
4. **Backlog** (`planning/backlog.md`; large file - grep for the item, read only around it):
   - approved: add "-> agent plan approved YYYY-MM-DD: `agent-plans/approved/<YYMMDD>-<slug>/`"
     to the item. Keep `#agent` (the review log already stops a re-plan).
   - deferred: add "-> agent plan deferred YYYY-MM-DD until <condition>".
   - rejected, drop the item: remove it (and its `#agent` tag with it).
   - rejected, keep: remove the `#agent` tag and add "-> agent plan rejected YYYY-MM-DD:
     <reason>".
   - deferred parts of an approved plan: add one backlog line each, under the same block.
5. **Decisions.** Only if the review settled a real design question (for example "Atera and
   Xenium readers stay separate"), propose an entry for `planning/decisions.md` and add it if
   the maintainer agrees. Routine outcomes do not go there.
6. Confirm the filing in one line, then move to the next plan.

## Step 4: Summary

1. Table: plan, outcome, one-line rationale, next action.
2. The `/implement` commands for the approved plans, in a sensible order, each with its
   prerequisite (branch to start from, merge it waits for):
   `/implement @agent-plans/approved/<YYMMDD>-<slug>/report-<slug>.md`
3. Deferred plans that were re-reviewed and the reminders that are still open.
4. Add a session log entry to `.log/log.md` (global CLAUDE.md format).
5. Offer the next step: **`/implement-insitupy-plans`** implements all runnable approved plans as a
   batch (one Sonnet subagent per plan in its own worktree, then an Opus review and the
   bookkeeping); pass plan folder names to limit it. For a single plan, `/implement @<report>`.
   Both are new commands: start them after this one, not inside this session's reasoning.
