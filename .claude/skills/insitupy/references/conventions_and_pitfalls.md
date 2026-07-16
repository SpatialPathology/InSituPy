# Conventions and pitfalls

Non-obvious behavior that isn't clear from a single docstring in isolation. Each item names
where to re-verify it if the codebase has moved on.

## `cells_layer` vs `layer` - two different concepts share similar names

- `cells_layer` selects *which segmentation layer* of `MultiCellData` to use (e.g. `"main"`,
  `"baysor"`, `"proseg"`). `None` means "use `cells.main_key`".
- `layer` (when a function also takes it, e.g. `pl.spatial`, `pp.normalize_and_transform`)
  means an `AnnData.layers[...]` key (e.g. raw vs. normalized counts) - unrelated to cell
  segmentation.
- Both appear side by side in several signatures (`ispy.pl.spatial`,
  `ispy.pp.normalize_and_transform`) - easy to pass one where the other is meant. Check with
  `introspect.py functions <module>` if a call has both.

## Construction: never build `InSituData` directly on a saved project path

- `InSituData(path=...)` raises `InSituDataConstructorPathError` if `path` already contains a
  saved `.ispy` project. Always load saved projects with `InSituData.read(path)`.
- `data.from_insitudata` (bool property) tells you whether the object is disk-backed by a saved
  project (`True`) vs. freshly constructed / read from raw platform output (`False`). Several
  `load_*()` methods branch on this - see `insitupy/_core/data.py` around `from_insitudata`.
- Reference: `tests/test_construction_guard.py`.

## Lazy loading mainly applies to *saved projects*, not raw platform reads

- After `InSituData.read(saved_project_path)`, `images`/`cells`/`transcripts`/`annotations`/
  `regions` may not be populated yet. Call `data.load_images()`, `data.load_cells()`,
  `data.load_transcripts()`, `data.load_annotations()`, `data.load_regions()` explicitly before
  assuming they're there. `InSituExperiment` has the same `load_*()` methods, applied across
  all samples.
- By contrast, raw-platform readers (`ispy.io.read_xenium()`, etc.) and the bundled
  `ispy.datasets.*` sample-data loaders (which just call `read_xenium()` under the hood) already
  populate `cells` and `images` eagerly as part of the read - no `load_cells()`/`load_images()`
  needed afterward. Only transcripts may still be dask-backed depending on `transcript_mode`.
- Images and transcripts stay dask-backed (lazy) even after "loading" in most cases; only
  cell-table access (`.cells.table`) forces materialization into memory (h5ad-backed AnnData).

## `inplace: bool = False` is the house convention

- Nearly every mutating method (`crop`, filtering, etc.) across `InSituData`, `CellData`,
  `MultiCellData`, `BoundariesData`, `ImageData`, `ShapesData`, `SpatialUnitsData` defaults to
  `inplace=False` and **returns a new object** rather than mutating. Pass `inplace=True`
  explicitly if you want the mutating behavior. Verify per-call with `introspect.py source
  <module> <Class>.<method>` since a few older/experimental functions (e.g. some
  `insitupy.preprocessing.anndata` functions) mutate their `adata` argument in place instead -
  check the docstring's "Returns" line, don't assume.

## `filter_cells` / `filter_genes` take exactly one criterion at a time

- `pp.filter_cells(data, min_counts=..., min_genes=..., max_counts=..., max_genes=..., mask=...)`
  raises `ValueError` if **more than one** of those five arguments is not `None` - unlike plain
  `sc.pp.filter_cells`, they are not combinable in a single call. To apply several thresholds,
  call it multiple times in sequence, or build a combined boolean `mask` yourself and pass only
  `mask`. Same constraint applies to `pp.filter_genes`. Reference:
  `insitupy/preprocessing/experiment.py::filter_cells`.

## Public names are often re-exported, not defined, at the module you'd expect

- Many public functions (most of `insitupy.datasets`, much of `insitupy.io`) are implemented in
  an internal submodule (e.g. `insitupy/datasets/datasets.py`) and only re-exported through
  their package's `__init__.py`. Both `mcp__insitupy__get_function_source` and
  `introspect.py functions/classes/source` need the *defining* module, not just any module the
  name is importable from - passing the package name alone silently comes up empty (or, for the
  MCP tool, can error with "module has no attribute ...").
- Fix: run `introspect.py whereis <name>` (or `mcp__insitupy__search_codebase` for
  `"def <name>"`) first to find the file that actually defines it, then target that file.

## Writing a new `insitupy.pp`/`insitupy.tl` function that accepts `InSituExperiment | InSituData`

Follow the existing dispatch pattern rather than inventing a new one - copy the shape of
`insitupy/preprocessing/experiment.py::filter_cells`:
1. Branch with `insitupy._core._checks._is_experiment(data)`.
2. If it's an experiment, iterate `data.iterdata()`; otherwise treat `data` itself as the single
   sample (`zip([None], [data])` is the pattern used in `filter_cells`).
3. Resolve the working cell layer via `insitupy.containers._utils._get_cell_layer(cells=xd.cells,
   cells_layer=cells_layer)` rather than reaching into `xd.cells.table` directly, so the
   `cells_layer=None` -> `main_key` convention keeps working.
- Add a matching test in `tests/` named after the function/area (see `list_test_files`), not a
  generic catch-all test file.

## Lazy-loaded AnnData from `InSituExperiment.table[...]`

- `exp.table["layer"]` (the concatenated cross-sample table accessor) can return an `AnnData`
  built via `anndata.experimental.read_lazy`, where `.var` is a `Dataset2D` (xarray-backed), not
  a plain pandas `DataFrame`.
- To load only `.var` into memory without touching the expression matrix `.X`:
  ```python
  adata.var.to_memory()   # -> pandas DataFrame; does not load X
  ```
  `Dataset2D.to_memory()` reads only the `var/` zarr group, which is the idiomatic way to get a
  real DataFrame out of it (`pd.DataFrame(adata.var)` also works but isn't the idiomatic call).

## `MultiCellData.main_key` / `set_main()`

- You cannot delete the current main layer (`KeyError`) without promoting another layer first.
- `set_main(key)` **silently no-ops** if `key` isn't a known layer - it does not raise. Check
  `data.cells.keys()` first if a `set_main()` call doesn't seem to take effect.

## `normalize_and_transform` assumes raw integer counts by default

- `assert_integer_counts=True` (the default) raises if the count matrix isn't integer-valued -
  a common trip-up when re-running on already-normalized or externally-corrected data. Preview
  with `pl.test_transformations()` first, or pass `assert_integer_counts=False` deliberately
  when counts are legitimately non-integer.
- Normalization stores intermediates in `adata.layers["counts"]` (raw) and
  `adata.layers["norm_counts"]` (normalized) before applying the log1p/sqrt transform.

## Deprecated code paths - two different kinds

- Live but deprecated wrapper functions (e.g. `pl.plot_spatial`, `pl.plot_overview`,
  `pl.plot_cellular_composition`, `pl.plot_colorlegend`, `tl.register_images`) still work but
  just call through to the current function (`spatial`, `overview`, `cellular_composition`,
  `colorlegend`, `im.register_images_standalone`) - use the current name in new code.
- `insitupy._deprecated` is dead code (commented out), kept only as historical reference - not
  a runnable API at all. Don't import from it.

## Two distinct "metadata"

- `InSituExperiment.metadata` is a cross-sample `pd.DataFrame` (one row per sample).
- `InSituData.metadata` is a per-sample `dict` (method info, history, uids, cropping history).
  Don't confuse the two when a task says "add metadata".

## Areas under active churn (double-check before trusting memory)

- `insitupy.spatialdata` (convert_to/from_spatialdata) and `cells.boundaries` /
  `nucleus_to_cell_map` have had repeated breaking fixes recently. Check
  `git log --oneline -- insitupy/spatialdata insitupy/containers/boundaries_data.py` and the
  matching `tests/test_spatialdata_*.py` / `tests/test_boundaries_data_save.py` before relying
  on previously-seen behavior in this area.

## Tests are the most reliable usage examples

- The suite has 60+ files under `tests/`, each usually named after the exact area it covers
  (`test_<area>.py`). Before guessing a kwarg or call pattern, check
  `mcp__insitupy__list_test_files` or `introspect.py grep "<name>"` (it searches both
  `insitupy/` and `tests/` by default) for a real call site.

## Windows / environment (already covered in the repo's CLAUDE.md - not repeated here)

Environment setup, which tests to run, and PowerShell portability rules live in the project
`CLAUDE.md`, not in this skill - don't duplicate them.
