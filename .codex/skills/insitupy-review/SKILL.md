---
name: insitupy-review
description: "Self-contained, InSituPy-specific severity-graded review of your working changes before a PR. Prints a PR-ready summary to paste into the pull request (required by AI_POLICY.md)."
---
<!-- AUTO-GENERATED from .claude/commands/review.md by tools/sync_commands.py - do not edit here; edit the canonical file and re-run the sync. -->

# /review - InSituPy contributor code review

You are performing a rigorous, InSituPy-specific review of the changes on the current branch so the
contributor can (per AI_POLICY.md) paste a summary of what it found into their PR. This command is
**self-contained**: it carries its own full rubric and does not depend on any private/user-level
review agent. It works on a fresh clone, on any agent.

## Fallback contract
If the `insitupy` MCP server is available in this session, prefer its live tools
(`get_function_source`, `get_data_model`, `search_codebase`, ...) to verify how the changed code uses
the API. If it is not available, verify against the checked-out source and the installed package
(`python -c "import insitupy, inspect; ..."`, or read the files). Never rely on a remembered API
shape - confirm it against the live tools or the source.

## 1. Determine the review scope
- If `$ARGUMENTS` names a base ref (`main`, `origin/main`, a commit, or `A..B`), review that range.
- Otherwise default to the branch's own changes:
  - BASE = `git merge-base HEAD origin/main` (fall back to `main` if `origin/main` is absent).
  - Review `git diff BASE...HEAD` **plus** uncommitted changes (`git diff`, and `git status` for
    new files).
- List the changed files first. If the diff is empty, say so and stop.

## 2. Load the InSituPy pitfalls checklist
Read `.claude/skills/insitupy/references/conventions_and_pitfalls.md` (committed; on every clone)
and check the diff against **every** item - these are non-obvious behaviours no signature or
docstring reveals. Highest-value checks (the file is authoritative and may have grown; use it, not
this excerpt, as the full list):
- `inplace=False` is the house convention - a mutating call whose return value is ignored is a bug.
- `cells_layer` (which segmentation layer) vs `layer` (an `AnnData.layers` key) - easy to swap.
- `filter_cells` / `filter_genes` take **exactly one** criterion per call (raise otherwise).
- Public names are often re-exported, not defined, where you'd expect - verify the defining module.
- Never construct `InSituData(path=...)` on a saved project - use `InSituData.read(path)`.
- A new `pp`/`tl` function taking `InSituExperiment | InSituData` must follow the existing dispatch
  pattern (`_is_experiment` branch, `iterdata()`, `_get_cell_layer`).

## 3. Review dimensions
Find problems in the diff:
- **Correctness** - logic/edge-case errors, mutation-vs-return bugs, API misuse (checked against the
  pitfalls file + live source).
- **Security** - unsafe path/file handling, deserialization, injection, secrets.
- **Performance** - needless materialization of lazy dask/AnnData, O(n^2) over cells/transcripts,
  large-array copies - respect the lazy-loading conventions.
- **Maintainability** - clarity, dead code, naming that fights the surrounding code.
- **Tests, at altitude** - does the change warrant a test of a **real failure mode**? Flag missing
  tests for real logic; do NOT ask for change-detector tests or tests that stub the unit under test.
  Trivial glue can be verified manually. (Mirror the test philosophy in CLAUDE.md.)

Report **only problems**. Do not describe what works. Do not flag style that already matches the
surrounding code.

## 4. Grade and report
Group findings by severity; one finding = one concrete problem (do not bundle). Omit empty sections.

## Critical - breaks correctness/security/data; must fix before merge
### [C1] <title>
- **File:** <path:line>
- **Issue:** <what is wrong>
- **Impact:** <what breaks>
- **Fix:** <concrete suggestion>

## Major - likely bug, performance cliff, or a missing test for real logic
## Minor - maintainability/clarity; safe to defer
## Needs maintainer judgement - design-intent / architectural / subtle-security calls you cannot
resolve from the diff alone. Surface them here for a human maintainer to weigh in - do not
silently drop them.

## 5. PR-ready summary (required)
End with this block; the contributor pastes it into the PR (AI_POLICY.md):

> **AI-assisted review summary**
> - Reviewed: <files / diff range>
> - Found & fixed: <one line each, or "none">
> - Knowingly left open: <one line each, or "none">
> - Needs maintainer judgement: <one line each, or "none">

Write no report file - a fresh clone has no `.log/`. Keep the review in this conversation and in the
summary above.
