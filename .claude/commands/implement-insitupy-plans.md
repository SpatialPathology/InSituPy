---
description: Implement approved agent plans as a batch - one Sonnet subagent per plan in its own worktree, then an Opus review of each branch and the bookkeeping. Follow-up to /walkthrough-insitupy-plans.
argument-hint: "[plan folder names or slugs] - optional; defaults to all plans in agent-plans/approved/"
model: claude-opus-5-5
allowed-tools: Read, Glob, Grep, Edit, Write, PowerShell, Bash, Agent, SendMessage, AskUserQuestion, ToolSearch, mcp__insitupy__*
---

# /implement-insitupy-plans - batch-implement approved agent plans

You are the **orchestrator**. Sonnet subagents implement, one plan each, in their own git
worktree. You choose and order the batch, review every result, and do all shared bookkeeping.
Running this command is the maintainer's request to use subagents, but confirm the concrete batch
(step 2) before spawning any.

## Scope

$ARGUMENTS

Empty: every plan folder in `agent-plans/approved/`. Otherwise only the named folders
(`<YYMMDD>-<slug>`) or slugs.

## Step 1: Collect

1. `agent-plans/` must exist at the repo root (local link, see `.claude/setup-notes.md`). If it is
   missing, stop and say so.
2. Read each plan in scope fully, including its `## Review` section (it overrides the rest). Note:
   base branch, target (`main`, backport to a `release/*` branch), parts and which of them are due
   now (e.g. "Part A now, Part B after a merge"), prerequisites (a branch merge, a manual step by
   the maintainer such as a probe run), and the files and functions it touches.
3. Check drift: `git fetch`, then whether each prerequisite is met (`git branch -r --merged
   origin/main`, `git log`). A plan or part whose prerequisite is not met is not runnable: list it
   in the summary with what it waits for, do not start it.
4. Build a conflict map: two plans that edit the same function or file region run one after the
   other (the second branches from the first's branch, or waits for its merge - say which).
   Generated files (`insitupy/_ai/skill/reference/`) are regenerated on each branch; on a later
   conflict, regenerate again instead of merging by hand. `poetry.lock` changes: one plan at a time.

## Step 2: Confirm the batch

1. Show a table: plan, part, branch name, base ref, group (parallel group number or order),
   prerequisites, conflicts. Branch names follow the repo's history (`fix/...` for bug fixes,
   `feat/...`, `chore/...`).
2. Preflight, before asking:
   - **Test environment:** the python of the environment where InSituPy is installed. Take it from
     your user config or memory (machine-specific, never written into this repo). If unknown, ask.
     Check it with `<python> -c "import insitupy, sys; print(sys.executable)"`.
   - **Commit signing:** if `git config commit.gpgsign` is true, make sure the signing key is
     available now (user config says how). A subagent cannot type a passphrase.
3. Ask with AskUserQuestion: run as proposed (recommended) / a subset / a different order. Spawn
   nothing before the answer.

## Step 3: Spawn

For each runnable plan of the current group, in parallel, one `Agent` call:

- `subagent_type: "general-purpose"`, `model: "sonnet"`, `isolation: "worktree"`.
- `prompt`: the template below with every `<...>` filled in. `<ABS_REPORT_PATH>` is the absolute
  path of the report in this checkout's `agent-plans/approved/` (the subagent's worktree has no
  `agent-plans/`). If a subagent reports it cannot read that path, send the report text with
  `SendMessage`.
- `SendMessage` (used here and in step 4) is a deferred tool: load it once with `ToolSearch`
  `select:SendMessage` before the first use. Subagents stay addressable by the name or ID their
  `Agent` call returned.

Start the next group when the current one has been reviewed (step 4) and its branches are the
base the next group needs.

### Subagent prompt template

```
You implement one approved plan in your own git worktree. Work only inside this worktree.

Setup
1. Read `CLAUDE.md` and `.claude/implement-protocol.md` in your worktree and follow the protocol.
2. Plan report (read only, outside your worktree): <ABS_REPORT_PATH>
   Its `## Review` section overrides the rest. Scope for this run: <SCOPE>.
3. Branch: run `git fetch origin`, then `git switch -c <BRANCH> <BASE_REF>`. Check with
   `git status` that you are on <BRANCH> and the tree is clean.
4. Test environment: <PYTHON>. From the worktree root, first run
   `<PYTHON> -c "import insitupy; print(insitupy.__file__)"`. The path must lie inside this
   worktree; if it does not, stop and report (tests would run against another checkout). Run
   tests as `<PYTHON> -m pytest <test files>` from the worktree root.

Your role: batch subagent (overrides CLAUDE.md and the protocol where they conflict)
- Commit on <BRANCH> with messages in the style of `git log`. No push, no PR, no tags, no other
  GitHub writes. Never bypass commit signing; if a commit fails on signing, stop and report.
- Do not read or write `.log/`, `planning/` or `agent-plans/` (except reading the report). Do not
  edit the backlog, release notes or session log; propose release-note lines in your final
  message instead.
- Do not spawn subagents. Do not ask questions. On an open design question, a contradiction
  between report and code, or a test failure you cannot fix within scope: stop and report.
- Lint only the files you changed if the report asks for it; no codebase-wide cleanup.

Final message - exactly these headings:
## Result: done | partial | stopped
## Branch and commits   (branch, base, output of `git log --oneline <BASE_REF>..HEAD`)
## Files changed
## Tests                (each command run, pass/fail counts, tests added and why)
## Acceptance criteria  (each: met / not met / n.a., with evidence)
## Deviations from the plan
## Open questions and blockers
## Release-note lines   (proposed wording, or "none")
## Follow-ups           (out-of-scope findings)
```

## Step 4: Review each branch (as results arrive)

1. Read the subagent's final message. Find its worktree with `git worktree list`.
2. Diff: `git -C <worktree> diff <BASE_REF>...<BRANCH>`; check that `git -C <worktree> status` is
   clean.
3. Review the diff against:
   - the report, its `## Review` section and its acceptance criteria;
   - the rubric of `.claude/commands/review.md` sections 2-4 (both pitfalls checklists, the review
     dimensions, severity grading).
4. Verify, do not trust the subagent's report: rerun its targeted tests from the worktree root
   (same `insitupy.__file__` check); run `python tools/generate_skill_reference.py --check` if
   public API changed; run `poetry check --lock --strict` if `pyproject.toml` dependencies changed.
5. Critical or major findings: send concrete fix instructions to the same subagent with
   `SendMessage`, then review again (at most two rounds, then ask the maintainer). Minor findings:
   fix them yourself in the worktree or list them. Design-intent questions: collect for the
   maintainer.
6. Verdict per plan: **ready**, **ready with notes**, or **not ready** (with the reason).

## Step 5: Bookkeeping (orchestrator only, right after each review, one plan at a time)

Subagents never touch these files, so there are no concurrent edits. Use Edit for edits and
PowerShell for moves.

- **Plan folder:** ready -> move `agent-plans/approved/<YYMMDD>-<slug>/` to
  `agent-plans/implemented/<YYMMDD>-<slug>/`. Partial or not ready -> leave it in `approved/`.
- **`agent-plans/review-log.md`:** update the plan's row - outcome `implemented` (or `partial`:
  what is missing), date, branch, new location.
- **`planning/backlog.md`:** mark or remove the completed item per the global CLAUDE.md; note
  missing parts and follow-ups.
- **`planning/release-notes.md`:** add the (edited) release-note lines under the in-progress
  version (see "Logging, reports, and backlog" in `CLAUDE.md`).
- **`.log/log.md`:** one entry per plan (global CLAUDE.md format).

## Step 6: Hand back

1. Summary table: plan, branch, result, review verdict, tests, open points.
2. Per ready branch, the PR-ready summary block of `.claude/commands/review.md` section 5 for the
   PR body.
3. Ask before any push or PR, per branch. Backports to a `release/*` branch come after the merge to
   `main` and go through their own PR; list them as next actions, do not start them.
4. Not-run plans and parts, with what they wait for. Questions for the maintainer.
5. Worktrees: offer `git worktree remove <path>` for branches that are pushed or abandoned; the
   branches stay.
