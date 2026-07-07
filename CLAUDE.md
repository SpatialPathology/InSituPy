# InSituPy

InSituPy (`insitupy-spatial` on PyPI) is a Python framework for **histology-guided,
multi-sample analysis of single-cell spatial transcriptomics data**, currently focused on
10x Genomics _Xenium In Situ_ data. License: BSD-3-Clause. Requires Python >=3.12.

## Data model

The package is built around a two-level hierarchical data structure:

- **`InSituData`** — a single sample/slide. Integrates all modalities: `cells`
  (`MultiCellData` → named `CellData` layers, each wrapping a scanpy `AnnData` table plus
  segmentation `boundaries`), `images` (`ImageData`, lazy dask arrays), `transcripts`,
  `annotations`, `regions`, and `units` (`SpatialUnitsData`).
- **`InSituExperiment`** — aggregates multiple `InSituData` instances with a sample-level
  `metadata` DataFrame, enabling cross-sample analysis. Subscript access returns an
  `InSituData` (`experiment[i]`).

Use the `insitupy` MCP server (below) for an always-current view of the data model; prefer it
over this summary if they ever conflict.

## Public API surface

`import insitupy` exposes `InSituData`, `InSituExperiment`, and submodule shorthands:

- `insitupy.io` — readers (`read_xenium`, `read_visium`, `read_qupath`, …)
- `insitupy.pp` — preprocessing (normalize, filter, pseudobulk, …)
- `insitupy.tl` — analysis tools (DGE, distance, neighbors, registration, …)
- `insitupy.pl` — plotting (spatial, umap, volcano, overview, …)
- `insitupy.im` — image utilities
- `insitupy.utils`, `insitupy.datasets`, `insitupy.interactive`, `insitupy.spatialdata`

## Repository layout

- `insitupy/` — the package source
- `tests/` — pytest suite (`tests/data/` fixtures; `conftest.py` shared fixtures)
- `tools/mcp_server/` — source of the `insitupy` MCP server
- `docs/` — Sphinx documentation
- `pyproject.toml` — single source of truth for version and dependencies (poetry-core build)

## Environment & commands

- **Reading and planning need nothing installed.** The source tree is on disk and the
  `insitupy` MCP server runs in its own environment, so code exploration and `/plan` work
  without InSituPy installed in the active session environment.
- **Running tests requires InSituPy installed** in the active environment — but do not install
  it yourself. Run `pytest` from that environment. If a test run fails because InSituPy isn't
  importable (`ModuleNotFoundError`), surface that to the user and ask them to install it
  (`pip install -e ".[dev]"`, or their conda/uv equivalent) instead of treating it as a code
  failure or retrying in another environment. The specific environment name/path is
  developer-/machine-specific — keep it in your own user config, not in this shared file.
- **Run only targeted tests** relevant to the changed method/module, not the full suite — the
  full suite is slow and has known pre-existing failures that produce misleading output (a
  `-x` stop can look like a regression when it isn't). Find the covering test files (e.g. those
  referencing the changed class) and run just those. Run the full suite only when explicitly
  asked or when a change is broad enough to warrant it.
- **Linting:** ruff is configured in `pyproject.toml` (rule sets `E,W,F,I,UP,B,NPY`).

## Platform notes

InSituPy is developed across Windows, macOS, and Linux — keep shell commands portable. On
Windows, prefer the PowerShell tool over Bash, and mind PowerShell 5.1 limitations: no
`&&`/`||` chaining (use `;`/`if`), no ternary/`??`, and a UTF-16 default file encoding (pass
`-Encoding utf8` when writing files other tools must read). Machine-specific tool quirks (e.g.
a broken Bash PATH on a particular setup) belong in your own user config, not in this shared
file.

## Tools available — `insitupy` MCP server

An MCP server named `insitupy` exposes the source tree for exploration. **Tools are deferred:
call `ToolSearch` with `select:mcp__insitupy__<name>` to load a schema before first use.**

- Introspection: `list_modules`, `list_classes`, `list_functions`, `get_class_info`,
  `get_function_source`, `get_docstring`, `read_source_file`, `search_codebase`,
  `list_test_files`
- Curated overviews: `get_data_model`, `get_public_api`, `get_io_formats`,
  `get_storage_format`, `get_plotting_api`, `get_preprocessing_api`, `get_tools_api`,
  `get_workflow_guide`, `get_datasets_guide`, `get_interactive_guide`, `get_images_api`,
  `get_spatialdata_api`, `get_result_types`

Prefer these MCP tools over blind file searching when answering questions about InSituPy
internals.

## Logging, reports, and backlog

Follow the layout defined in the global `~/.claude/CLAUDE.md`:

- Reports → `.log/reports/YYMMDD/<short-task-title>/report-<short-task-title>.md`
- Session log → `.log/log.md` (append via Edit, **never** overwrite) after any task that
  changed files
- Backlog → `.log/backlog.md`

### `.log/` is not tracked by git — symlink required on each machine

`.log/` is excluded from version control and backed up via a private cloud-synced folder.
On a fresh clone, recreate `.log/` as a symlink to that folder. See `.claude/setup-notes.md`
for instructions.

## Coding workflow

This project uses a two-command plan→implement workflow with per-phase model tiering. The
report written by `/plan` is the durable handoff: implementation can follow immediately or in
a later session with identical results, because all behavioral configuration lives here, in
the agent files, and in the commands — not in the planning conversation.

- **`/plan <goal>`** — pinned to Opus, read-only, ends by writing a self-contained report to
  `.log/`. Makes no code edits.
- **`/implement @<report>`** — pinned to Sonnet, executes the report. The report is the only
  per-run input.

The session model default is `opusplan` (set in `.claude/settings.json`): entering plan mode
in a same-sitting run gives Opus for planning and drops to Sonnet for execution; a deferred
implement-only run never enters plan mode and stays on Sonnet. The commands' own `model`
frontmatter pins each phase explicitly regardless of the session default.

### Agents and escalation

- **Implement directly by default. Ask before spawning any subagent.**
- `@code-reviewer` (user-level) — structured severity-graded reviews saved to `.log/reports/`.
- `@opus-consultant` (user-level) — hard architecture / security / correctness / design-intent
  questions via a structured Opus Report → recommendation/clarify contract. Use for
  **escalations that need the structured contract** (e.g. a Reviewer escalate milestone).
- Native **`/advisor`** toggle — use for lightweight "Sonnet is stuck" moments during
  execution where a full structured Opus Report would be overkill. Lower friction; keeps Opus
  on-call without leaving the session.
