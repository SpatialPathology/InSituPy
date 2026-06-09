# Cross-Sample Concatenated Table

This tutorial introduces the `.table` attribute on `InSituExperiment`, a zarr-backed
concatenated `AnnData` that spans all samples in an experiment. It enables
memory-efficient access to the full dataset for analysis steps such as
cross-sample clustering, dimensionality reduction, and cell-type annotation —
and provides a clean way to transfer results back to the individual per-sample
objects.

:::{note}
This feature is experimental and may change in future versions.
:::

## Workflow overview

<center><img src="../../_static/img/insituexperiment_table_architecture.svg" width="900"/></center>

<br>

The workflow has three phases:

1. **Build** — `build_table()` concatenates all per-sample AnnData objects and
   writes the result to `{experiment_path}/tables/{cells_layer}.zarr`
   (e.g. `tables/main.zarr`).
2. **Analyse** — access the concatenated table via `.table[<layer>]`, run any
   scanpy / scvi-tools workflow on it, and write results back to the zarr store.
3. **Import** — `import_from_table()` transfers selected obs columns and obsm
   keys from the zarr table back into the individual per-sample AnnData objects
   so results are available for spatial visualisation.

This is the zarr-native counterpart to the
{doc}`AnnData workflow <InSituPy_Anndata_workflow>` (`to_anndata()` /
`import_from_anndata()`). The key difference is that `.table` stays on disk
and is read lazily, avoiding the need to hold the full concatenation in RAM
after the analysis step.

---

## Setup

```python
from pathlib import Path
from insitupy import InSituExperiment

# Load a saved experiment (must have a path for build_table to write to)
exp = InSituExperiment.read("path/to/my_experiment")
exp
```

---

## Phase 1: Building the concatenated table

### Basic build (in-memory concatenation)

```python
exp.build_table(
    label_col="uid",           # Metadata column that identifies each sample
    obsm_keys="spatial",       # Keep spatial coordinates (default)
    make_obs_names_unique=True,# Prefix each cell name with "{sample_index}-"
)
```

This creates a per-layer zarr table at
`{experiment_path}/tables/{cells_layer}.zarr` (e.g. `tables/main.zarr`). The
build metadata (label column, method, layer name) is stored inside the zarr
store under `uns["_insitupy_build_params"]` — no separate JSON file is written.

### Selecting what to include

Use the `obs_keys`, `var_keys`, `obsm_keys`, and `layer_keys` arguments to
control what is written to the zarr. For example, to include raw counts and
spatial coordinates:

```python
exp.build_table(
    obsm_keys=["spatial"],
    layer_keys=["counts"],     # Include the raw counts layer
    obs_keys="all",            # Include all existing obs columns
)
```

To add experiment-level metadata columns (e.g. `condition`, `batch`) to
each cell's obs row:

```python
exp.build_table(
    metadata_keys=["condition", "batch"],  # Columns from exp.metadata
)
```

### Gene panels that differ across samples

There is no join strategy to choose. `build_table()` always stores the
**union** of all genes on disk and reconstructs the correct **inner** gene set
automatically each time you read `.table`:

- `exp.table[<layer>]` returns the genes shared by **all** samples.
- `view.table[<layer>]` returns the genes shared by only the **view's**
  samples — recovering genes that are common within the subset even when they
  are absent from some other sample in the full experiment.

No fill values are ever surfaced through `.table`. Pass `min_shared_genes` to
get a warning when very few genes are shared across *all* samples:

```python
exp.build_table(
    min_shared_genes=100,      # Warn if fewer than 100 genes are shared by all samples
)
```

### Rebuilding an existing table

```python
exp.build_table(overwrite=True)
```

---

## Memory-efficient build with `concat_on_disk`

For large experiments that do not fit in RAM, use
`method="concat_on_disk"`. This mode uses
{func}`anndata.experimental.concat_on_disk` to stream each sample's saved
`table.h5ad` directly to the output zarr store without loading all samples
simultaneously.

**Prerequisites:**
- The experiment must have been saved with `saveas()` (i.e. each
  `InSituData` must have an on-disk path and a saved `cells/` directory).

```python
exp.build_table(
    method="concat_on_disk",
    make_obs_names_unique=True,
)
```

**Differences from `"in_memory"` mode:**

| Feature | `"in_memory"` | `"concat_on_disk"` |
|---|---|---|
| Obs name prefix | `"{index}-{cell}"` (numeric index) | `"{uid}-{cell}"` (label value) |
| Filter args (`obs_keys`, `var_keys`, …) | Supported | Not supported — raises `ValueError` |
| Metadata columns | Supported | Not supported |
| Requires datasets saved | No | Yes |

---

## Phase 2: Accessing and analysing the table

### Accessing `.table`

```python
# `.table` is a dict-like accessor — index it by cells-layer name:
tbl = exp.table["main"]   # lazily-loaded AnnData for the "main" layer
# exp.table.keys()        # list available layer names
tbl
```

If `build_table()` has not been called yet, indexing `.table` returns `None`
and emits a `UserWarning`.

### Running a scanpy workflow

The concatenated table is a standard `AnnData` object — any scanpy or
scvi-tools function works directly on it.

```python
import scanpy as sc

tbl = exp.table["main"]   # reconstructed inner gene set — no fill values

# Normalise and log-transform
sc.pp.normalize_total(tbl, target_sum=1e4)
sc.pp.log1p(tbl)

# Dimensionality reduction
sc.pp.highly_variable_genes(tbl, batch_key="uid")
sc.tl.pca(tbl, use_highly_variable=True)
sc.pp.neighbors(tbl, n_neighbors=15, n_pcs=30)
sc.tl.umap(tbl)

# Clustering
sc.tl.leiden(tbl, resolution=0.5, key_added="leiden_integrated")

# Inspect the result
sc.pl.umap(tbl, color=["uid", "leiden_integrated"])
```

### Saving results back to the zarr table

`import_from_table()` reads from the **raw** on-disk store, so write new obs
columns or obsm keys back to it directly — not via `exp.table[...]`, which
returns a reconstructed gene subset that would not round-trip the union layout:

```python
import anndata as ad

zarr_path = Path(exp.path) / "tables" / "main.zarr"
tbl = ad.read_zarr(zarr_path)          # raw union table (holds fill values)

# Run the analysis itself on the fill-free `exp.table["main"]`, then attach the
# resulting obs columns / obsm keys (e.g. clustering, embeddings) onto `tbl`.

tbl.write_zarr(zarr_path)              # Write updated table back to disk
```

---

## Phase 3: Importing results back into per-sample objects

`import_from_table()` reads the zarr table and transfers selected columns/keys
into the individual per-sample `AnnData` objects:

```python
exp.import_from_table(
    obs_columns=["leiden_integrated"],   # Clustering result
    obsm_keys=["X_umap"],               # UMAP embedding
)
```

After importing, call `exp.save()` to persist the changes to disk:

```python
exp.save()
```

Results are now available in every per-sample `InSituData` for spatial
visualisation:

```python
# Plot integrated clusters spatially for sample 0
exp.data[0].cells.table.obs["leiden_integrated"].value_counts()

# Plot UMAP embedding coloured by leiden cluster for the first sample
import scanpy as sc
sc.pl.umap(exp.data[0].cells.table, color="leiden_integrated")
```

---

## Working with filters and views

`InSituExperimentView` — created by `exp.filters.apply()` or direct slicing —
has its own `.table` accessor. It does **not** merely row-slice the parent
table: it reconstructs the correct gene set for the view's samples — the inner
join over only the view's datasets — and row-filters to those samples. This
recovers genes shared within the subset even when they are absent from some
other sample in the full experiment:

```python
# Apply a filter defined on the experiment
view = exp.filters.apply("high_quality")

# View's table: inner-over-view genes, restricted to the view's rows
view_tbl = view.table["main"]
print(f"View covers {view_tbl.n_obs} cells from {len(view)} samples")
```

This is read-only, and a view cannot call `build_table()` — build on the
parent experiment, then read the per-view result via `view.table[...]`.

---

## Full example

```python
from pathlib import Path
import scanpy as sc
import anndata as ad
from insitupy import InSituExperiment

# 1. Load experiment
exp = InSituExperiment.read("path/to/my_experiment")

# 2. Build zarr table (large experiment → use concat_on_disk)
exp.build_table(
    method="concat_on_disk",
    make_obs_names_unique=True,
)

# 3. Load and analyse
zarr_path = Path(exp.path) / "tables" / "main.zarr"
tbl = ad.read_zarr(zarr_path)

sc.pp.normalize_total(tbl, target_sum=1e4)
sc.pp.log1p(tbl)
sc.pp.highly_variable_genes(tbl, batch_key="uid")
sc.tl.pca(tbl, use_highly_variable=True)
sc.pp.neighbors(tbl, n_neighbors=15, n_pcs=30)
sc.tl.umap(tbl)
sc.tl.leiden(tbl, resolution=0.5, key_added="leiden_integrated")

# 4. Write results back to zarr
tbl.write_zarr(zarr_path)

# 5. Transfer back to per-sample objects
exp.import_from_table(
    obs_columns=["leiden_integrated"],
    obsm_keys=["X_umap"],
)
exp.save()

# 6. Visualise spatially
sc.pl.umap(exp.table["main"], color=["uid", "leiden_integrated"])
```

---

## API reference

| Method / attribute | Description |
|---|---|
| `exp.build_table(...)` | Build `tables/{cells_layer}.zarr` (e.g. `main.zarr`); supports `in_memory` and `concat_on_disk` |
| `exp.table[<layer>]` | Lazily load the concatenated AnnData (inner gene set over all samples) |
| `exp.import_from_table(obs_columns, obsm_keys)` | Transfer columns/keys from the zarr table back to per-sample objects |
| `view.table[<layer>]` | View's table: inner gene set over the view's samples, row-filtered to them |

```{eval-rst}
.. seealso::

   :doc:`InSituPy_Anndata_workflow`
       The in-memory counterpart using ``to_anndata()`` and ``import_from_anndata()``.
```
