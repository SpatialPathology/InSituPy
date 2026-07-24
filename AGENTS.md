# AGENTS.md

Orientation for any AI coding agent working in this repository (Claude Code, Codex, Cursor,
or otherwise). InSituPy (`insitupy-spatial` on PyPI) is a Python framework for
histology-guided, multi-sample analysis of single-cell spatial transcriptomics data.

## Prefer the live MCP server if present

If the `insitupy` MCP server is available in this session, prefer its live tools over anything
written here or in any static reference file - they resolve against the actually installed
source and never go stale. See `tools/mcp_server/README.md` for setup.

## Writing InSituPy analysis code

Use the bundled **`insitupy-api` skill** - it teaches the data model, the read -> preprocess -> tools
-> plot -> save workflow arc, and links to full API references:

- Already in this clone: `insitupy/_ai/skill/` (point a skill-aware agent at it directly).
- Not yet installed for your agent: `insitupy install-skill` (after `pip install
  insitupy-spatial`; see `README.md`).
- No skill loader / no MCP (plain web chat): paste the repo-root `llms.txt`.

The skill is versioned (see its frontmatter). If a function or argument you need is missing
from it, **suspect a stale skill before concluding it does not exist** - check the installed
version (`python -c "import insitupy; print(insitupy.__version__)"`) and re-run `insitupy
install-skill --force` (or re-fetch `llms.txt` / the skill ZIP) if it's out of date.

## Modifying the InSituPy package source

Use the contributor **pitfalls skill** at `.claude/skills/insitupy-pitfalls/` (mirrored to
`.cursor/skills/insitupy-pitfalls/` and `.codex/skills/insitupy-pitfalls/`) - it catalogs
non-obvious API conventions and traps not visible from a single docstring or signature. Two
canonical files, both under `.claude/skills/insitupy-pitfalls/reference/`:
`conventions_and_pitfalls.md` (API traps for any caller; **also copied into the shipped
`insitupy-api` skill**, so keep it free of repo-only material) and `contributing_to_insitupy.md`
(source-modification conventions; repo-only, never shipped). Edit them there, not in a mirrored
copy (`tools/sync_commands.py` regenerates the others). `CONTRIBUTING.md` ("Which files are
canonical, and which are generated") has a diagram of the whole pipeline.

Before opening a PR: read `AI_POLICY.md`, then run the review workflow and paste its summary
into the PR (see `CONTRIBUTING.md` - `/review` in Claude Code and Cursor, the
`insitupy-review` skill in Codex).

Maintainer-oriented `/plan` and `/implement` commands also exist (Claude Code only) - see
`CONTRIBUTING.md`.
