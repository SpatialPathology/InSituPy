---
name: insitupy-pitfalls
description: Non-obvious API conventions and pitfalls for the InSituPy spatial transcriptomics package (this repo) - things the codebase's own MCP server, docstrings, and signatures don't say outright. Use whenever writing, reviewing, or debugging code that imports insitupy, or modifies the insitupy package source itself.
---
<!-- AUTO-GENERATED from .claude/skills/insitupy-pitfalls/SKILL.md by tools/sync_commands.py - do not edit here; edit the canonical file and re-run the sync. -->

# InSituPy

InSituPy (`insitupy-spatial` on PyPI) is a Python framework for histology-guided,
multi-sample analysis of single-cell spatial transcriptomics data, currently focused on 10x
Genomics Xenium.

For the data model, typical workflows, storage format, and current function signatures, use
the `insitupy` MCP server's curated tools (`get_data_model`, `get_workflow_guide`,
`get_storage_format`, `get_io_formats`, `get_plotting_api`, `get_preprocessing_api`,
`get_tools_api`, `get_spatialdata_api`, `get_interactive_guide`, `get_datasets_guide`,
`get_result_types`, `search_codebase`, `get_function_source`, `get_docstring`, ... - see this
repo's `CLAUDE.md` for the full list). Those resolve live against the checked-out source, so
nothing here duplicates them - trust them over anything remembered from a previous session. If
the `insitupy` MCP server is not available in your agent, the pitfalls file below is
self-contained; verify live API shapes against the installed package or the checked-out source
instead.

What this skill adds on top, split by audience:

- `reference/conventions_and_pitfalls.md` - API traps that apply to anyone *calling* insitupy:
  naming collisions between similarly-named parameters, silent no-ops, and mutual-exclusivity
  constraints between arguments. Read it before writing or reviewing non-trivial code against
  this package. This file is also shipped to end users inside the `insitupy-api` skill, so keep
  it free of repo-only material (test paths, `git` commands, MCP tool names).
- `reference/contributing_to_insitupy.md` - conventions for *modifying the package source*: the
  `pp`/`tl` dispatch pattern, locating a name's defining module, areas under active churn, and
  using the test suite as usage examples. Repo-only; never shipped.
