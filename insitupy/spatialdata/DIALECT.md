# InSituPy SpatialData naming dialect

This document specifies the element-naming convention used by
`insitupy.spatialdata.convert_to_spatialdata()` when writing an `InSituData` or
`InSituExperiment` to a `SpatialData` object/zarr store. It is the single source of truth for
the dialect; `convert_from_spatialdata()` (and any external tool reading InSituPy's exports)
should be written against this spec.

The store also carries a versioned, machine-readable copy of the essentials at
`sdata.attrs["insitupy_spatialdata_dialect"]` (see below) so a reader can detect "this store is
InSituPy dialect, version N" without pattern-matching element-name strings.

## Reading

`insitupy.spatialdata.read_spatialdata(path)` (file-path convenience wrapper) and
`insitupy.spatialdata.convert_from_spatialdata(sdata)` (operates on an already-loaded
`SpatialData` object) are the public reader entry points - a true inverse of
`convert_to_spatialdata()`. Both auto-detect every modality-naming detail (pixel sizes,
RGB-ness, per-cell-layer boundaries, per-unit-layer tables, ...) from the store itself and this
dialect descriptor; no caller-supplied keys are needed. They return a bare `InSituData` for a
single-sample store, or an `InSituExperiment` of `InSituData` objects for a multi-sample store.
Both raise `ValueError` if `sdata.attrs` carries no `insitupy_spatialdata_dialect` key (a
foreign/labels-native store InSituPy did not write - out of scope for these functions) or if the
descriptor's version is not the one this InSituPy version reads.

The returned object has no backing project directory - call `.saveas(path)` to persist it as a
`.insitupy` project before `.save()` can be used.

`insitupy.spatialdata.convert_table_from_spatialdata(sdata, cells_layer, covered_labels=None)` is
a separate reader entry point for the `TABLES.<layer>` element (see below) - it is **not** part of
`convert_from_spatialdata()`'s reconstruction and does not attach anything to the returned
`InSituData`/`InSituExperiment`. The concatenated table is a disk-derived artifact in InSituPy's
own data model (never an in-memory `InSituExperiment` attribute), and reading it back preserves
that: it returns a plain `AnnData`, not a modification of the reconstructed experiment.

## Version

Current dialect version: **3** (`insitupy._constants.SPATIALDATA_DIALECT_VERSION`).

Version 3 added the `TABLES.<layer>` concatenated-table element
(`InSituExperiment.build_table()`'s union table), the `tables` derived modality
(`insitupy._constants.SPATIALDATA_DERIVED_MODALITIES`), and
`convert_table_from_spatialdata()` (see below).

Version 2 added the reserved `_insitupy_seg_mask_value` obs column on `CELLS.<key>.table` and
the `CELLS.<key>.nucleus_map` element (see below), plus `samples`/`slide_id`/`sample_id` sample
identity in the dialect descriptor. No backward-compatible reading of version-1 stores is
implemented (beta line; no version-1 stores are known to exist outside test runs).

## Key grammar

```
{SAMPLE.<uid>..}?<MODALITY>.<locator>[.<locator>...]
```

- The `SAMPLE.<uid>..` prefix (note the trailing double dot) is present **only** when converting
  an `InSituExperiment`. A bare `InSituData` produces un-prefixed keys (e.g. `IMAGES.nuclei`,
  `CELLS.main.table`) - kept short for readability as napari layer names.
- `<uid>` is the sample's `InSituExperiment` metadata `uid`.
- `<MODALITY>` is always upper-cased (`IMAGES`, `CELLS`, `UNITS`, `TRANSCRIPTS`, `ANNOTATIONS`,
  `REGIONS`).
- `<locator>` segments are joined with `.`; `.` and `-` inside a locator name are replaced with
  `_` to satisfy SpatialData's key-naming rules.
- `TRANSCRIPTS` has no locator (there is at most one transcripts element per sample).

## Emitted patterns

Every `MODALITY.locator` pattern the writer actually emits, and the SpatialData element type it
becomes:

| Pattern | Element type | Notes |
|---|---|---|
| `IMAGES.<name>` | `Image2DModel` | One entry per named image channel. |
| `CELLS.<key>.table` | `TableModel` | Cell expression table for cell layer `<key>`. Carries a reserved `_insitupy_seg_mask_value` obs column when boundaries exist (see below). |
| `CELLS.<key>.circles` | `ShapesModel` (circles) | Unit-radius circles at cell centroids. |
| `CELLS.<key>.circles_sized` | `ShapesModel` (circles) | Only if `cell_area` is available in `obs`; radius derived from area. |
| `CELLS.<key>.boundaries.<name>` | `Labels2DModel` | Segmentation mask, e.g. `boundaries.cells` / `boundaries.nuclei`. |
| `CELLS.<key>.nucleus_map` | `TableModel` (optional) | One row per nucleus; present only when the cell layer's boundaries carry a populated `nucleus_to_cell_map` (multinucleated-cell support, Xenium v2.0+). See below. |
| `UNITS.<key>.table` | `TableModel` | Omics table for spatial units layer `<key>`. |
| `UNITS.<key>.shapes` | `ShapesModel` (polygons) | The units' own polygon geometries - not synthesized, unlike cell circles. |
| `TABLES.<layer>` | `TableModel` | `InSituExperiment.build_table()`'s concatenated union table for cells layer `<layer>`. Experiment-level - **no `SAMPLE.` prefix**, even in a multi-sample export. Present only when `build_table()` was called for that layer (opt-in via `convert_to_spatialdata(..., include_concat_tables=False)` to omit even when built). See below. |
| `TRANSCRIPTS` | `PointsModel` | Omitted entirely when `include_transcripts=False`. |
| `ANNOTATIONS.<key>` | `ShapesModel` (polygons) | One entry per annotation key. |
| `REGIONS.<key>` | `ShapesModel` (polygons) | One entry per region key. |

Cells and units are structurally parallel (both are table + geometry pairs) but use different
locator names for the geometry element (`circles` vs `shapes`) to signal that cell circles are
synthesized from centroids while units carry their own real polygons.

### `_insitupy_seg_mask_value` obs column

`CellData.is_synced` requires `table.obs_names == boundaries.cell_names` in the same order, but
the writer only exports the boundary *raster* (pixel values = the true segmentation mask ids,
not necessarily contiguous or 1-based - real Xenium data is not) and the table's own `cell_id`
(the string cell name, used to link the table to `circles`, not to the raster). Without an
explicit link, a reader could only guess `seg_mask_value = arange(1, n+1)` - wrong for real,
non-contiguous segmentation. `_insitupy_seg_mask_value` closes this gap: a reserved obs column
on `CELLS.<key>.table`, present whenever the cell layer has boundaries, holding each cell's true
segmentation mask value in `obs` row order. Absent when the cell layer has no boundaries, or on
a hypothetical pre-version-2 store (in which case the reader falls back to the same
`arange(1, n+1)` assumption, now as an explicit, logged fallback rather than the only path).

### `CELLS.<key>.nucleus_map`

`BoundariesData.nucleus_to_cell_map` / `nucleus_count` support multinucleated cells (Xenium
v2.0+): several nuclei can map to one cell. Rather than adding more per-cell obs columns (which
would mix per-cell and per-nucleus cardinality into one table), this mapping gets its own
dedicated table - one row per nucleus - that annotates the cell layer's `boundaries.nuclei`
`Labels2DModel` raster via SpatialData's own `region`/`instance_key` mechanism (the same
table-annotates-labels idiom already used for the main cells table -> `circles` link, applied a
second time to a previously-unannotated raster):

- `region` = the `CELLS.<key>.boundaries.nuclei` element key.
- `instance_key` -> obs column `nucleus_label`: the 1-indexed nucleus mask value (matches the
  raster's own pixel values).
- Data column `cell_id`: the parent cell's name, joins to `CELLS.<key>.table.obs_names`.

`nucleus_count` is **not** stored - it is derived on read via a group-by on `cell_id`. This is a
deliberate normalization: InSituPy's native `nucleus_count` comes from an independent Xenium
zarr column (not computed from `nucleus_to_cell_map`), so the two could in principle disagree;
deriving on read avoids storing a second, possibly-inconsistent field. Absent entirely for the
ordinary (non-multinucleated) case - `nucleus_to_cell_map`/`nucleus_count` reconstruct as `None`,
matching `BoundariesData`'s own "not available, assume 1:1" semantics.

Not every nucleus mask necessarily gets a row: Xenium marks nuclei that were never assigned to a
cell (orphan nuclei) with an out-of-range `cell_index`, and a map can go stale after boundaries
are filtered without a following `.sync()`. Both are excluded from the exported table rather than
resolved to a bogus `cell_id` (see `insitupy.utils.utils.is_valid_boundary_index`, the shared
predicate also used by the napari label-alignment path).

### `TABLES.<layer>`

`InSituExperiment.build_table()` writes an on-disk, per-cells-layer concatenated union `AnnData`
(outer join across samples, `X` filled with 0 for genes a given sample didn't measure) plus a
gene-presence record that lets readers reconstruct the correct *inner* gene set on demand - see
`TableAccessor`/`ViewTableAccessor` in `insitupy/experiment/data.py`. `TABLES.<layer>` carries
this union table, plus the presence record, into the SpatialData export unchanged:

- `uns["_insitupy_gene_presence"]`, `uns["_insitupy_presence_labels"]`,
  `uns["_insitupy_table_format_version"]`, `uns["_insitupy_build_params"]` (including
  `label_col`, `method`, `cells_layer`, and `make_obs_names_unique`) all pass through the zarr
  round trip verbatim - the same keys `build_table()` writes to `tables/<layer>.zarr` locally.
- `region` is a **list** of every contributing sample's `CELLS.<layer>.circles` key (a
  multi-region table, per SpatialData's table-annotates-multiple-elements idiom) - not a single
  key like the per-sample tables use.
- Each row's `region` obs value (categorical, one of the list above) and `cell_id` (the
  `instance_key`) correctly link that row back to its origin sample's real cell: `cell_id` holds
  the *original*, pre-concatenation cell name (recovered by stripping the
  `-{label_col value}` suffix `anndata.concat(..., index_unique="-")` appends when
  `make_obs_names_unique=True`, or used as-is when `False`), matching that sample's
  `CELLS.<layer>.circles` index exactly. This is what makes the element meaningful to an
  external SpatialData-aware viewer, not merely round-trippable by InSituPy's own reader.

**Opt-in, not always written.** A layer is exported only if `build_table()` was called for it
(`experiment.table.keys()` is non-empty for that layer); `convert_to_spatialdata(...,
include_concat_tables=False)` skips it even when built. A layer whose on-disk table is stale
relative to the current experiment (e.g. a covered sample was removed, so its circles element
won't be exported this time) is skipped with a warning rather than partially exported, since a
`region` list must reference only elements that actually exist in the store.

**Reading it back:** `insitupy.spatialdata.convert_table_from_spatialdata(sdata, cells_layer,
covered_labels=None)` reapplies `TableAccessor`'s inner-over-covered reconstruction (unmodified)
to the stored element - `covered_labels=None` reconstructs the full-experiment table
(`exp.table[cells_layer]`); passing a subset of labels reconstructs the inner-over-that-subset,
row-filtered table (`view.table[cells_layer]`). Not part of `convert_from_spatialdata()`'s
`InSituData`/`InSituExperiment` reconstruction - see "Reading" above.

## `sdata.attrs["insitupy_spatialdata_dialect"]`

For an `InSituExperiment` (multi-sample) export, sample identity is keyed by `uid`:

```python
{
    "insitupy_spatialdata_dialect": {
        "version": 3,
        "modalities": ["cells", "units", "images", "transcripts", "annotations", "regions", "tables"],
        "sample_prefix_pattern": "SAMPLE.<uid>..",
        "samples": {
            "<uid>": {"slide_id": "...", "sample_id": "..."},
            ...
        },
    }
}
```

For a bare `InSituData` export (no `SAMPLE.` prefix - only one sample, so no `uid` keying is
needed), `slide_id`/`sample_id` are flat keys instead:

```python
{
    "insitupy_spatialdata_dialect": {
        "version": 3,
        "modalities": ["cells", "units", "images", "transcripts", "annotations", "regions", "tables"],
        "sample_prefix_pattern": "SAMPLE.<uid>..",
        "slide_id": "...",
        "sample_id": "...",
    }
}
```

Arbitrary per-sample `metadata` dicts, `method_name`, `method_params`, and `pixel_size` are
**not** persisted here - an explicit, documented gap, not an oversight. Only `uid` (free from the
key prefix) plus `slide_id`/`sample_id` round-trip.

This is namespaced under a single key (not written at the top level of `attrs`) because
`convert_from_spatialdata()` forwards the whole `sdata.attrs` dict into each reconstructed
`InSituData`'s `method_params`; namespacing keeps the dialect descriptor from polluting that
dict with unrelated top-level keys.

## Zarr format

spatialdata `0.8.0` combined with zarr-python `3.x` writes **zarr v3** stores (every group is
`zarr.json`, no `.zgroup`/`.zarray`). `SpatialData.write()` has no format-selection argument -
the format follows the installed zarr-python major version, which is why `pyproject.toml` pins
`zarr` to `(>=3.2.1,<4.0.0)`. Exported InSituPy SpatialData stores are therefore zarr v3 and
require a v3-capable reader.

This is unrelated to InSituPy's own `.insitupy` on-disk project format, which may still be zarr
v2 on disk from older exports - that legacy read path (`containers/_zarr_compat.py`,
`tests/test_zarr_v2_read_smoke.py`) is retained and unaffected by this spec.

## `spatialdata` version floor

`pyproject.toml` bounds the optional `spatialdata` extra to `>=0.8.0,<0.9.0`. `sdata.attrs` has
existed since spatialdata `0.3.0`, so nothing here strictly requires `0.8.0` - the floor is
raised for reliability: all development and testing happens against `0.8.0`, and `0.8.0 +
zarr-python 3.x` is the verified zarr-v3-producing combination, whereas an older floor risks
dragging in zarr-python 2.x and zarr v2 stores.

## Transcript export cost

Measured on a real 42.6M-row transcript table (`xenium_human_breast_cancer`, ~160k cells):
`PointsModel.parse(..., sort=True)` took **18.93 s**. spatialdata's parse internally treats the
`feature_name` column as having "unknown categories," forcing an extra dask pass to determine
them. Pre-converting the column to a **known** categorical
(`df["feature_name"].astype("category").cat.as_known()`) before calling `parse` drops this to
**8.41 s** (~2.25x faster) - this optimization is applied unconditionally for dask-backed
transcript tables. For experiments where even this is too slow, pass
`include_transcripts=False` to `convert_to_spatialdata()` to skip transcript export entirely.

## Transcript `feature_name` dtype invariant

`InSituData.transcripts["feature_name"]` is a **known categorical** on export (see
"Transcript export cost" above). The invariant and who enforces it:

- **Export** pre-computes known categories (`.cat.as_known()`) for `PointsModel.parse` speed.
- **Write** (`insitupy/containers/io.py::_save_transcripts`) forces a uniform `int32`
  dictionary index across partitions for every categorical column, so a partitioned store
  whose partitions carry independently-sized dictionary indices still serializes against one
  schema.
- **Importers** (the dialect reader and `convert_from_foreign_spatialdata`) must never cast
  the column to `str` - doing so discards the categorical dtype and makes the lazy transcript
  viewer's `.unique()` O(rows) instead of O(#categories).

The raw Xenium reader (`read_xenium`) leaves `feature_name` as `string[pyarrow]`; that is
intentional and needs no normalization (its `.unique()` is already cheap - measured 539 genes
in 0.38 s on a real bundle).
