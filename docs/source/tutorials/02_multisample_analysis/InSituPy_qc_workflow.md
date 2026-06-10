# Per-Sample QC Workflow

Quality control in InSituPy follows a **two-step pattern** that mirrors the scanpy
idiom:

1. **Compute** — `pp.calculate_qc_metrics(exp)` writes per-cell QC columns
   (`total_counts`, `n_genes_by_counts`, …) into each sample's `cells.table.obs`.
2. **Summarize** — `exp.qc_summary()` reads those columns and returns a
   per-dataset `DataFrame` (one row per sample) with medians, means, and cell
   counts.

Keeping compute and summarize separate means the two steps can be run
independently, visualised at the single-cell level with `pl.plot_qc_metrics`, and
combined with the filter system to tag and subset low-quality samples.

:::{note}
`pp.calculate_qc_metrics` mirrors `sc.pp.calculate_qc_metrics` and is the
**only** QC compute step. The legacy method `exp.calculate_qc_metrics()` is
deprecated and will be removed in a future release.
:::

---

## Setup

```python
import insitupy.pp as pp
import insitupy.pl as pl
from insitupy import InSituExperiment

exp = InSituExperiment.read("path/to/my_experiment")
exp
```

---

## Step 1 — Compute per-cell metrics

`pp.calculate_qc_metrics` iterates over all samples and writes QC columns into
each sample's cell table **in place**. No value is returned.

```python
pp.calculate_qc_metrics(exp)
```

This adds `total_counts`, `n_genes_by_counts`, `log1p_total_counts`,
`log1p_n_genes_by_counts`, `pct_counts_in_top_*_genes`, and related columns to
`adata.obs` (and gene-level columns to `adata.var`) for every sample.

**Restricting to a non-default cell layer:**

```python
# For data with an alternative segmentation (e.g. ProSeg)
pp.calculate_qc_metrics(exp, cells_layer="proseg")
```

The `cells_layer` argument is forwarded to each sample's `MultiCellData`. When
omitted, the main layer is used.

---

## Step 2 — Summarize per dataset

Once per-cell metrics have been computed, call `exp.qc_summary()` to aggregate
them into a per-sample table:

```python
df = exp.qc_summary()
df
```

The returned `DataFrame` is indexed by `uid` and has the following columns:

| Column | Description |
|---|---|
| `cells_layer` | The cell layer the metrics were read from |
| `n_cells` | Number of cells in that sample |
| `median_total_counts` | Median UMI count per cell |
| `mean_total_counts` | Mean UMI count per cell |
| `median_n_genes_by_counts` | Median number of detected genes per cell |
| `mean_n_genes_by_counts` | Mean number of detected genes per cell |

`df.attrs["cells_layer"]` records the resolved layer name so you can tell at a
glance which layer was summarised.

**Passing `cells_layer` must match the compute step:**

```python
# Metrics were computed on 'proseg' → summarise the same layer
df = exp.qc_summary(cells_layer="proseg")
```

If the required columns are absent (i.e. `pp.calculate_qc_metrics` was not run
for that layer), a `ValueError` is raised naming the layer and pointing to the
compute step.

---

## Visualizing per-sample distributions

Inspect the cell-level QC distribution for a single sample using
`pl.plot_qc_metrics`:

```python
# Inspect QC metrics for the first sample
pl.plot_qc_metrics(exp.data[0])
```

To loop over all samples and show distributions side by side:

```python
for i, (_, dataset) in enumerate(exp.iterdata()):
    print(f"Sample {i}: {dataset.uid}")
    pl.plot_qc_metrics(dataset)
```

`plot_qc_metrics` accepts optional `counts_thresh` and `genes_thresh` arguments
to draw vertical threshold lines on the histograms.

---

## Writing the summary into experiment metadata

Pass `add_to_metadata=True` to append the five summary columns into
`exp.metadata`:

```python
df = exp.qc_summary(add_to_metadata=True)
exp.metadata.head()
```

For a non-main layer the columns are suffixed with ` (<layer>)` so results from
different layers coexist without collision:

```python
exp.qc_summary(cells_layer="proseg", add_to_metadata=True)
# Adds: 'n_cells (proseg)', 'median_total_counts (proseg)', …
```

Once the columns are in `exp.metadata`, call `exp.save()` to persist them to disk.

---

## Filtering samples by QC criteria

Combine `qc_summary` with the `FilterManager` to define named QC-based sample
subsets.

**1. Add a pass/fail column from the summary:**

```python
df = exp.qc_summary(add_to_metadata=True)

# Mark samples with at least 500 cells and median ≥ 200 UMIs as passing
exp.metadata["qc_pass"] = (
    (exp.metadata["n_cells"] >= 500) &
    (exp.metadata["median_total_counts"] >= 200)
).map({True: "pass", False: "fail"})
```

**2. Create a named filter:**

```python
exp.filters.create(
    by="qc_pass",
    include="pass",
    key="qc_pass",
    note="Samples with ≥500 cells and median_total_counts ≥200",
)
```

**3. Work with the filtered subset:**

```python
# Lazy view — no data is copied
view = exp.filters.view("qc_pass")
print(f"QC-passing samples: {len(view)} / {len(exp)}")

# Summarise only the passing samples
view.qc_summary()
```

Persist both the metadata column and the filter definition with `exp.save()`.

---

## Working with `InSituExperimentView`

`qc_summary()` is inherited by `InSituExperimentView` and operates on only the
view's samples:

```python
view = exp.filters.view("qc_pass")
view_df = view.qc_summary(add_to_metadata=True)
```

`add_to_metadata=True` on a view writes to the **view's own in-memory metadata**
and does **not** propagate back to the parent experiment. To populate the parent,
call `qc_summary(add_to_metadata=True)` on `exp` directly.

---

## Full example

```python
import insitupy.pp as pp
import insitupy.pl as pl
from insitupy import InSituExperiment

# 1. Load experiment
exp = InSituExperiment.read("path/to/my_experiment")

# 2. Compute per-cell QC metrics
pp.calculate_qc_metrics(exp)

# 3. Summarize at dataset level
df = exp.qc_summary(add_to_metadata=True)
print(df)

# 4. Inspect per-sample distributions for the first sample
pl.plot_qc_metrics(exp.data[0])

# 5. Tag samples and create a filter
exp.metadata["qc_pass"] = (
    (exp.metadata["n_cells"] >= 500) &
    (exp.metadata["median_total_counts"] >= 200)
).map({True: "pass", False: "fail"})

exp.filters.create(
    by="qc_pass",
    include="pass",
    key="qc_pass",
    note="Samples with ≥500 cells and median_total_counts ≥200",
)

# 6. Work with the passing subset
view = exp.filters.view("qc_pass")
print(f"QC-passing samples: {len(view)} / {len(exp)}")

# 7. Persist
exp.save()
```

---

## API reference

| Method / function | Description |
|---|---|
| `pp.calculate_qc_metrics(exp, cells_layer=None)` | Compute per-cell QC metrics (in place) for all samples |
| `exp.qc_summary(cells_layer=None, add_to_metadata=False)` | Aggregate per-cell metrics into a per-dataset DataFrame |
| `pl.plot_qc_metrics(dataset, counts_thresh=None, genes_thresh=None)` | Plot per-sample QC histograms with optional threshold lines |
| `exp.filters.create(by, include/exclude, key, note)` | Create a named boolean mask from a metadata column |
| `exp.filters.view(key)` | Lazy `InSituExperimentView` of the selected samples |

```{eval-rst}
.. seealso::

   :doc:`InSituPy_filter_workflow`
       Create named filter masks from metadata columns, produce views and independent copies.

   :doc:`InSituPy_table_workflow`
       Build a cross-sample concatenated zarr table for integrated analysis.
```
