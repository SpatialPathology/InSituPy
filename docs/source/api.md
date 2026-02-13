# API

## Importing InSituPy
```{eval-rst}
.. code-block:: python

    import insitupy as isp
```

Individual submodules can then be imported like this:
```{eval-rst}
.. code-block:: python

    isp.dataclasses
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
.. module:: insitupy._core
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
.. module:: insitupy.experiment
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated/core_data_objects

    InSituExperiment.read
```

To generate a new `InSituExperiment` object, either from a configurations file or from histological regions, following functions are available:

```{eval-rst}
.. module:: insitupy.experiment
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated/core_data_objects

    InSituExperiment.from_config
    InSituExperiment.from_regions
```

To concatenate multiple `InSituExperiment` objects:

```{eval-rst}
.. module:: insitupy.experiment
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated/core_data_objects

    InSituExperiment.concat
```

Working with saved sample filters (including optional notes):

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

    # save and later load with filter applied
    exp.save()
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
.. module:: insitupy.dataclasses
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated/core_data_classes

    dataclasses.CellData
    dataclasses.MultiCellData
    dataclasses.BoundariesData
```

### Image data

```{eval-rst}
.. module:: insitupy.dataclasses
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated/core_data_classes

    dataclasses.ImageData
```

### Geometric data

```{eval-rst}
.. module:: insitupy.dataclasses
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated/core_data_classes

    dataclasses.ShapesData
    dataclasses.AnnotationsData
    dataclasses.RegionsData
```

The different data classes can be read using following functions:

```{eval-rst}
.. module:: insitupy.dataclasses
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated/core_data_classes

    dataclasses.read_celldata
    dataclasses.read_multicelldata
    dataclasses.read_shapesdata
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
    io.read_xenium
```

To read multiple datasets exported from QuPath into an `InSituExperiment` object, following functions can be used:

```{eval-rst}
.. module:: insitupy.io
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
    plotting.cellular_composition
    plotting.cell_abundance_along_axis
    plotting.cell_expression_along_axis
    plotting.volcano
    plotting.colorlegend
    plotting.overview
```
