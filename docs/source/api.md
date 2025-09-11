# API

```{eval-rst}
Import InSituPy as::

    import insitupy

.. module:: insitupy
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
.. module:: insitupy._core
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated/external_data

    read_qupath
    read_xenium
```

To read multiple datasets exported from QuPath into an `InSituExperiment` object, following functions can be used:

```{eval-rst}
.. module:: insitupy.experiment
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated/external_data

    read_qupath_project
```

---

## Plotting

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
