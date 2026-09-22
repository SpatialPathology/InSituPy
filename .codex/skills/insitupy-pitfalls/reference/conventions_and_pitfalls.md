<!-- AUTO-GENERATED from .claude/skills/insitupy-pitfalls/reference/conventions_and_pitfalls.md by tools/sync_commands.py - do not edit here; edit the canonical file and re-run the sync. -->

# Conventions and pitfalls

Non-obvious behavior that isn't clear from a single docstring in isolation. Each item names
where to re-verify it if the codebase has moved on.

## `cells_layer` vs `layer` - two different concepts share similar names

- `cells_layer` selects *which segmentation layer* of `MultiCellData` to use (e.g. `"main"`,
  `"baysor"`, `"proseg"`). `None` means "use `cells.main_key`".
- The `AnnData.layers[...]` key (e.g. raw vs. normalized counts) is a *different* concept,
  unrelated to cell segmentation. Its parameter name is not uniform: `pl.spatial` calls it
  `layer`, but `pp.normalize_and_transform` calls it **`adata_layer`** (there is no `layer`
  parameter there).
- In `ispy.pl.spatial`, `cells_layer` and `layer` appear side by side - easy to pass one where
  the other is meant. Check the signature with `inspect.signature(fn)` when a call has both, and
  don't assume the AnnData-layer arg is spelled `layer` everywhere.

## Construction: never build `InSituData` directly on a saved project path

- `InSituData(path=...)` raises `InSituDataConstructorPathError` if `path` already contains a
  saved `.ispy` project. Always load saved projects with `InSituData.read(path)`.
- `data.from_insitudata` (bool property) tells you whether the object is disk-backed by a saved
  project (`True`) vs. freshly constructed / read from raw platform output (`False`). Several
  `load_*()` methods branch on this - see `insitupy/_core/data.py` around `from_insitudata`.

## What "loading" means: `InSituData.read` loads all; only `InSituExperiment.read` defers

- `InSituData.read(path)` has `load_all=True` **by default**, so it eagerly loads every
  available modality (`images`/`cells`/`transcripts`/`annotations`/`regions`). You do *not* need
  to call `load_*()` afterward. Pass `load_all=False` to defer loading, then call
  `data.load_images()`, `data.load_cells()`, etc. on demand. (Verify: `InSituData.read` in
  `insitupy/_core/data.py`.)
- `InSituExperiment.read(path)` is the exception: it reads each dataset with `load_all=False`,
  so after reading an experiment the per-sample modalities are **not** in memory. Call the
  experiment-level `exp.load_cells()` / `exp.load_images()` / ... (applied across all samples)
  before assuming they're there.
- Raw-platform readers (`ispy.io.read_xenium()`, etc.) and the bundled `ispy.datasets.*` loaders
  (which call `read_xenium()` under the hood) populate `cells` and `images` as part of the read.
- Images and transcripts stay dask-backed (lazy) even after loading in most cases; only
  cell-table access (`.cells.table`) forces materialization into memory (h5ad-backed AnnData).

## `inplace: bool = False` is the house convention

- Container-level and geometric methods (`crop`, filtering, etc.) across `InSituData`,
  `CellData`, `MultiCellData`, `BoundariesData`, `ImageData`, `ShapesData`, `SpatialUnitsData`
  default to `inplace=False` and **return a new object** rather than mutating. Pass
  `inplace=True` explicitly for the mutating behavior.
- This is **not** a universal convention. The `insitupy.pp.*` preprocessing functions
  (`normalize_and_transform`, `filter_cells`, `cluster_cells`, ...) and the `assign_*` /
  `import_*` methods on `InSituData` mutate the object they are given in place and typically
  return `None` - there is no `inplace` parameter to flip. Don't write
  `data = ispy.pp.normalize_and_transform(data)` (you'll get `None`); just call it for its side
  effect.
- Verify per call rather than assuming: check the docstring's "Returns" line with `help(fn)`.

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
  their package's `__init__.py`. Any tool that needs the *defining* module comes up empty when
  handed the package name alone.
- Fix: ask the object where it lives instead of guessing.
  ```python
  import inspect
  import insitupy as ispy

  print(ispy.io.read_xenium.__module__)              # defining module
  print(inspect.getsourcefile(ispy.io.read_xenium))  # defining file
  ```

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
- `InSituData` also carries deprecated **warn-and-no-op** methods pulled in from
  `insitupy._core._deprecated`: `read_all`, `read_cells`, `read_images`, `read_annotations`,
  `read_regions`, `read_transcripts` (use the `load_*()` methods instead), plus
  `normalize_and_transform` / `reduce_dimensions` (use `insitupy.pp.*` instead). These only emit
  a `DeprecationWarning` and do nothing - a `data.read_cells()` call will silently leave `cells`
  unloaded. They are real attached methods, not dead code.

## Two distinct "metadata"

- `InSituExperiment.metadata` is a cross-sample `pd.DataFrame` (one row per sample). Reading it
  is free (no warning) and returns a **read-only** view: assigning to a column
  (`exp.metadata["group"] = ...`) or an attribute raises `InSituPyError` instead of silently
  failing. Use `exp.add_metadata_column(...)`, `exp.append_metadata(...)`, or
  `exp.set_metadata_values(...)` to change sample metadata, or `.copy()` for an editable frame.
- `InSituData.metadata` is a per-sample `dict` (method info, history, uids, cropping history).
  Don't confuse the two when a task says "add metadata".

## Subscripting, return types, and other 0.12 API-shape gotchas

Verified against `release/0.12.x`. These are shapes that look one way and behave another.

- **`experiment[i]` is a linked view, not an `InSituData`.** Subscripting an `InSituExperiment`
  returns an `InSituExperimentView` whose datasets are shared (not copied) with the parent; its
  `.cells` / `.images` / ... accessors print a message and return `None`. Use
  `experiment.data[i]` to get the underlying `InSituData`, or `experiment[i].copy()` for an
  independent `InSituExperiment`.
- **`assign_annotations` / `assign_regions` write to `obsm`, not `obs`, by default.** The result
  lands in `data.cells[layer].table.obsm["annotations"][key]` (resp. `["regions"]`) as one
  categorical column per key (`"unassigned"` where no polygon contains the cell). Pass
  `add_to_obs=True` to instead merge an `"annotations-{key}"` / `"regions-{key}"` column into
  `.obs`. `add_masks=True` only applies with `add_to_obs=True` and raises otherwise.
- **`images["missing_key"]` returns `None`, not a `KeyError`.** `ImageData.__getitem__` is a
  `dict.get`. Check membership explicitly before assuming an image is present.
- **`add_celldata(cd, key)` does not promote the layer to main.** `is_main` defaults to `False`,
  so the first layer you add is not automatically the main layer. Pass `is_main=True` (or call
  `set_main(key)` afterward). Note `set_main` silently no-ops on an unknown key (see above).
- **A non-inplace `crop()` returns a detached object with `uid = None`.** It belongs to no
  experiment; persist it standalone with `cropped.saveas(path)` before `save()` can be used, and
  `exp.add(cropped)` mints a fresh uid. (`copy()` and `crop(inplace=True)` keep the uid.)
- **`exp.add(xd)` for a dataset already present** warns and returns early; if you pass a
  conflicting `metadata=` for an already-added dataset it raises rather than silently dropping it
  (0.12). Add two distinct datasets rather than the same object twice.
- **`InSituData(metadata={...})` with a partial dict** is merged into the required skeleton
  (0.12), so `save()` / `crop()` no longer raise `KeyError: 'uids'` on a from-scratch object.
- **`show(return_viewer=True)` returns the napari viewer** (0.12; `exp.show` gained it too).
  Earlier versions returned `None` regardless.
- **`read_visium(...)` raises without a resolvable pixel size** (0.12) instead of silently
  falling back to 1.0 µm/px and leaving spot coordinates in pixels.
- **`register_images(...)` returns `None`.** It mutates the `InSituData` in place and writes
  registered images to `output_dir` (defaulting to next to `data.path` when omitted). There is
  no returned transformation object; the internal `ImageRegistration` engine holds `.T` etc.
- **`read_xenium(dataset_name=...)` is ignored** - that keyword does nothing; don't rely on it to
  name the dataset.
- **`.matrix` is a deprecated alias for `.table`** on `CellData` and `MultiCellData` (emits a
  `DeprecationWarning`). Use `.cells[layer].table` (or `.cells.table` for the main layer).
