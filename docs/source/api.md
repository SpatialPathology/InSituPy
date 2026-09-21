# API

## Importing InSituPy
```{eval-rst}
.. code-block:: python

    import insitupy as isp
```

Individual submodules can then be imported like this:
```{eval-rst}
.. code-block:: python

    isp.containers
    isp.io
    isp.plotting
```

---

## Core Data Objects

### Individual datasets

```{eval-rst}
.. module:: insitupy._core
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated/core_data_objects

    InSituData
```

Read a saved `InSituData` object with:

```{eval-rst}
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated/core_data_objects

    InSituData.read
```

### Handle multiple datasets

```{eval-rst}
.. module:: insitupy.experiment
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated/core_data_objects

    InSituExperiment
```

Read a saved `InSituExperiment` project with:

```{eval-rst}
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated/core_data_objects

    InSituExperiment.read
```

To generate a new `InSituExperiment` object, either from a configurations file or from histological regions, following functions are available:

```{eval-rst}
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated/core_data_objects

    InSituExperiment.from_config
    InSituExperiment.from_regions
```

To concatenate multiple `InSituExperiment` objects:

```{eval-rst}
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated/core_data_objects

    InSituExperiment.concat
```

Working with saved sample filters (experimental):

```{warning}
This filter workflow is currently **experimental** and may change in future releases.
```

```{eval-rst}
.. code-block:: python

    from insitupy import InSituExperiment

    exp = InSituExperiment.read("path/to/experiment")

    exp.filters.create(
        by="sample_id",
        include=["S01", "S02", "S05"],
        key="general_quality",
        note="Samples with overall good quality in total counts and morphology"
    )

    # overview table with selected/excluded counts and notes
    exp.filters.summary()

    # programmatic access to raw boolean masks
    exp.filters.masks()

    # full project save (datasets + metadata + colors + filters)
    exp.save()

    # dedicated partial-save helpers
    exp.save_metadata()
    exp.save_colors()
    exp.save_images(overwrite=False)

    # load later with filter applied
    exp2 = InSituExperiment.read("path/to/experiment", filter_key="general_quality")

    # detached subset (export workflow): exp_apply.path is None
    exp_apply = exp.filters.apply("general_quality")

    # linked lightweight view (in-place update workflow)
    exp_view = exp.filters.view("general_quality")
    exp_view.is_view            # True
    exp_view.applied_filters    # ["general_quality"]

    # Add another view filter; chain of applied filters is tracked
    exp_view2 = exp_view.filters.view("tumor_only")
    exp_view2.applied_filters   # ["general_quality", "tumor_only"]

    # view.save() updates only selected InSituData objects in-place
    # and does not overwrite experiment-level metadata/colors/filters
    exp_view2.save()

Notes:

- `exp.filters.apply(key)` returns a detached `InSituExperiment` subset (safe for `saveas()` export workflows).
- `exp.filters.view(key)` returns a lightweight linked view (`InSituExperimentView`) with path linkage preserved.
- Dataset identity is tracked via `uid` in metadata; filtered view indices are view-local and may differ from the parent experiment.
```

### Import data objects

Import the data objects like this:
```{eval-rst}
.. code-block:: python

    from insitupy import InSituData, InSituExperiment
```

---

## Core Data Classes

Data classes are used to store the different modalities.

### Cellular data

```{eval-rst}
.. module:: insitupy.containers
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated/core_data_classes

    containers.CellData
    containers.MultiCellData
    containers.BoundariesData
```

### Spatial units data

```{eval-rst}
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated/core_data_classes

    containers.SpatialUnitsData
    containers.MultiSpatialUnitsData
```

### Image data

```{eval-rst}
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated/core_data_classes

    containers.ImageData
```

### Geometric data

```{eval-rst}
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated/core_data_classes

    containers.ShapesData
    containers.AnnotationsData
    containers.RegionsData
```

The different data classes can be read using following functions:

```{eval-rst}
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated/core_data_classes

    containers.read_celldata
    containers.read_multicelldata
    containers.read_shapesdata
```

---

## Read external data

Following functions allow reading data from external sources, e.g. from an *Xenium In Situ* experiment or from [*QuPath*](https://qupath.github.io).
To read an individual dataset on can use following functions:
```{eval-rst}
.. module:: insitupy.io
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated/external_data

    io.read_qupath
    io.read_visium
    io.read_xenium
```

To read multiple datasets exported from QuPath into an `InSituExperiment` object, following functions can be used:

```{eval-rst}
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated/external_data

    io.read_qupath_project
```

---

## Plotting

Import the plotting submodule either as {code}`isp.plotting` or {code}`isp.pl`.

```{eval-rst}
.. module:: insitupy.plotting
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated/plotting

    plotting.spatial
    plotting.umap
    plotting.embedding
    plotting.cellular_composition
    plotting.cell_abundance_along_axis
    plotting.cell_expression_along_axis
    plotting.volcano
    plotting.dual_foldchange_plot
    plotting.colorlegend
    plotting.overview
```

---

## Preprocessing

Import the preprocessing submodule either as {code}`isp.preprocessing` or {code}`isp.pp`.

```{eval-rst}
.. module:: insitupy.preprocessing
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated/preprocessing

    pp.calculate_qc_metrics
    pp.filter_cells
    pp.filter_genes
    pp.normalize_and_transform
    pp.reduce_dimensions
    pp.cluster_cells
    pp.pseudobulk
    pp.calculate_mad_thresholds
```

---

## Tools

Import the tools submodule either as {code}`isp.tools` or {code}`isp.tl`.

```{eval-rst}
.. module:: insitupy.tools
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated/tools

    tl.dge
    tl.calc_distance_of_cells_from
    tl.pseudobulk_dge
    tl.register_images
```

---

## SpatialData integration

Import the spatialdata submodule as {code}`isp.spatialdata`. Convert between InSituPy and
[SpatialData](https://spatialdata.scverse.org/) objects (requires the `spatialdata` extra).

```{eval-rst}
.. module:: insitupy.spatialdata
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated/spatialdata

    spatialdata.convert_to_spatialdata
    spatialdata.convert_from_spatialdata
    spatialdata.convert_from_foreign_spatialdata
    spatialdata.read_spatialdata
```

---

## Image utilities

Import the image submodule either as {code}`isp.images` or {code}`isp.im`.

```{eval-rst}
.. module:: insitupy.images
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated/images

    im.read_image
    im.read_ome_tiff
    im.read_zarr
    im.write_ome_tiff
    im.write_zarr
    im.register_images_standalone
    im.apply_warp
    im.load_transformation_matrix
```

---

## Sample datasets

Import the datasets submodule as {code}`isp.datasets`. Sample datasets are downloaded on first
call and cached locally.

```{eval-rst}
.. module:: insitupy.datasets
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated/datasets

    datasets.list_downloaded_datasets
    datasets.xenium_human_breast_cancer
    datasets.visium_human_breast_cancer
```
