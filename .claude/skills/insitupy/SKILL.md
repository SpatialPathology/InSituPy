---
name: insitupy
description: Non-obvious API conventions and pitfalls for the InSituPy spatial transcriptomics package (this repo) - things the codebase's own MCP server, docstrings, and signatures don't say outright. Use whenever writing, reviewing, or debugging code that imports insitupy, or modifies the insitupy package source itself.
---

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
nothing here duplicates them - trust them over anything remembered from a previous session.

What this skill adds on top: `references/conventions_and_pitfalls.md` catalogs non-obvious
behavior that isn't visible from a single docstring or signature in isolation - naming
collisions between similarly-named parameters, silent no-ops, mutual-exclusivity constraints
between arguments, and areas of the codebase under active churn. Read it before writing or
reviewing non-trivial code against this package.
