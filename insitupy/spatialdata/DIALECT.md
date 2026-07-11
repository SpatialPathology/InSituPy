# InSituPy SpatialData naming dialect

This document specifies the element-naming convention used by
`insitupy.spatialdata.convert_to_spatialdata()` when writing an `InSituData` or
`InSituExperiment` to a `SpatialData` object/zarr store. It is the single source of truth for
the dialect; `convert_from_spatialdata()` (and any external tool reading InSituPy's exports)
should be written against this spec.

The store also carries a versioned, machine-readable copy of the essentials at
`sdata.attrs["insitupy_spatialdata_dialect"]` (see below) so a reader can detect "this store is
InSituPy dialect, version N" without pattern-matching element-name strings.

## Version

Current dialect version: **1** (`insitupy._constants.SPATIALDATA_DIALECT_VERSION`).

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
| `CELLS.<key>.table` | `TableModel` | Cell expression table for cell layer `<key>`. |
| `CELLS.<key>.circles` | `ShapesModel` (circles) | Unit-radius circles at cell centroids. |
| `CELLS.<key>.circles_sized` | `ShapesModel` (circles) | Only if `cell_area` is available in `obs`; radius derived from area. |
| `CELLS.<key>.boundaries.<name>` | `Labels2DModel` | Segmentation mask, e.g. `boundaries.cell_boundaries`. |
| `UNITS.<key>.table` | `TableModel` | Omics table for spatial units layer `<key>`. |
| `UNITS.<key>.shapes` | `ShapesModel` (polygons) | The units' own polygon geometries - not synthesized, unlike cell circles. |
| `TRANSCRIPTS` | `PointsModel` | Omitted entirely when `include_transcripts=False`. |
| `ANNOTATIONS.<key>` | `ShapesModel` (polygons) | One entry per annotation key. |
| `REGIONS.<key>` | `ShapesModel` (polygons) | One entry per region key. |

Cells and units are structurally parallel (both are table + geometry pairs) but use different
locator names for the geometry element (`circles` vs `shapes`) to signal that cell circles are
synthesized from centroids while units carry their own real polygons.

## `sdata.attrs["insitupy_spatialdata_dialect"]`

```python
{
    "insitupy_spatialdata_dialect": {
        "version": 1,
        "modalities": ["cells", "units", "images", "transcripts", "annotations", "regions"],
        "sample_prefix_pattern": "SAMPLE.<uid>..",
    }
}
```

This is namespaced under a single key (not written at the top level of `attrs`) because
`convert_from_spatialdata()` forwards the whole `sdata.attrs` dict into the reconstructed
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
