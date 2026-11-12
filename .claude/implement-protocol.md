You are implementing the plan report given below (or named in your prompt). This protocol is the
shared core of `/implement` (interactive, one plan) and of the subagents started by
`/implement-insitupy-plans` (batch, one plan each). The calling command adds your role: how to
escalate, whether you log, and what you report at the end. Where the role and this protocol
conflict, the role wins.

Conventions, environment rules and tool availability come from `CLAUDE.md`. The report carries only
the task. If `CLAUDE.md` is not already in your context (subagents), read it first.

## Protocol

1. **Read the report fully.** Confirm the goal, affected files and acceptance criteria. A
   `## Review` section (added when the maintainer reviewed an agent plan) overrides the rest of the
   report where they conflict, and its scope changes are part of the task. Stop and report instead
   of guessing if:
   - the report is missing or contradicts the current code;
   - its Verdict is "don't do" - then summarize the reasoning and implement nothing.
2. **Discover before writing.** Verify the report's file and line references against the current
   source; they may have drifted since planning (agent plans were written by reading the source
   only, without running anything). Prefer the `insitupy` MCP server when it is available
   (`mcp__insitupy__get_function_source`, `get_class_info`, `search_codebase`, `get_data_model`;
   load schemas via `ToolSearch` `select:` first), otherwise read the checked-out source. Match
   the conventions of the surrounding code.
3. **Implement incrementally** - smallest coherent edits first. Stay inside the report's scope; note
   anything worth doing outside it as a follow-up instead of doing it.
4. **Run targeted tests** for the changed method/module only, not the full suite, with `pytest` from
   the environment where InSituPy is installed (see `CLAUDE.md`). Before *adding* a test, apply the
   altitude check: a test must exercise the **real code path against a real failure mode**. Do not
   add change-detector tests (asserting the specific line/attribute you wrote) or tests that stub
   the unit under test; for trivial state/glue changes, verify manually and say so.
5. **Keep generated files in sync.** If you changed a public signature, a default or a Returns
   docstring, run `python tools/generate_skill_reference.py --check`. If it fails, regenerate with
   `python tools/generate_skill_reference.py` and include the result in the change. Use `--check`
   for this decision, not `tests/test_skill_reference_check.py` (that test only gates calls). If you
   changed a dependency declaration in `pyproject.toml`, follow "Dependency changes" in `CLAUDE.md`
   (`poetry lock`, same commit).
6. **Check the acceptance criteria** one by one: met, not met, or not applicable, each with the
   evidence (test name, command output, file).
7. **When genuinely stuck** on a design, security or correctness question, or when a test fails for
   a reason outside the report's scope, stop and surface it. Your role says how.
