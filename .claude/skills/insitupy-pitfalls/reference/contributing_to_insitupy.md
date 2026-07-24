# Contributing to the InSituPy source

Conventions that apply when you are **modifying the insitupy package itself**, working inside a
clone of this repository. Everything here assumes you have the source tree, the test suite, and
`git` available, and usually the `insitupy` MCP server too.

This file is deliberately **not** shipped in the `insitupy-spatial` wheel: only
`conventions_and_pitfalls.md` (the API traps that matter to anyone calling the library) is copied
into the user-facing `insitupy-api` skill by `tools/generate_skill_reference.py`. Keep
contributor-only material here so it does not leak into a published skill, where paths like
`tests/` and commands like `git log` are meaningless.

## Writing a new `insitupy.pp`/`insitupy.tl` function that accepts `InSituExperiment | InSituData`

Follow the existing dispatch pattern rather than inventing a new one - copy the shape of
`insitupy/preprocessing/experiment.py::filter_cells`:

1. Branch with `insitupy._core._checks._is_experiment(data)`.
2. If it's an experiment, iterate `data.iterdata()`; otherwise treat `data` itself as the single
   sample (`zip([None], [data])` is the pattern used in `filter_cells`).
3. Resolve the working cell layer via `insitupy.containers._utils._get_cell_layer(cells=xd.cells,
   cells_layer=cells_layer)` rather than reaching into `xd.cells.table` directly, so the
   `cells_layer=None` -> `main_key` convention keeps working.

Add a matching test in `tests/` named after the function/area (see `list_test_files`), not a
generic catch-all test file.

## Finding where a public name is actually defined

`conventions_and_pitfalls.md` covers the underlying trap (most public names are re-exported
through a package `__init__.py`, not defined there). The repo-side consequence is about tooling:
`mcp__insitupy__get_function_source` needs the *defining* module, not just any module the name is
importable from. Passing the package name alone silently comes up empty, or errors with
"module has no attribute ...".

- Fix: run `mcp__insitupy__search_codebase` for `"def <name>"` first to find the file that
  actually defines it, then target that file.

## Areas under active churn (double-check before trusting memory)

- `insitupy.spatialdata` (`convert_to`/`convert_from_spatialdata`) and `cells.boundaries` /
  `nucleus_to_cell_map` have had repeated breaking fixes recently. Check
  `git log --oneline -- insitupy/spatialdata insitupy/containers/boundaries_data.py` and the
  matching `tests/test_spatialdata_*.py` / `tests/test_boundaries_data_save.py` before relying
  on previously-seen behavior in this area.

## Tests are the most reliable usage examples

- The suite has 60+ files under `tests/`, each usually named after the exact area it covers
  (`test_<area>.py`). Before guessing a kwarg or call pattern, check
  `mcp__insitupy__list_test_files`, or grep `insitupy/` and `tests/` for a real call site.

## Windows / environment (already covered in the repo's CLAUDE.md - not repeated here)

Environment setup, which tests to run, and PowerShell portability rules live in the project
`CLAUDE.md`, not in this skill - don't duplicate them.
