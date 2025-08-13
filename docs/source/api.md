# API

```{eval-rst}

Import InSituPy as::

    import insitupy

.. module:: insitupy
```

## Core Data Objects

### Individual datasets

```{eval-rst}
.. module:: insitupy._core
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated

    InSituData
```

Read the `InSituData` object with:

```{eval-rst}
.. module:: insitupy._core
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated

    InSituData.read
```

### Handle multiple samples

```{eval-rst}
.. module:: insitupy.experiment
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated

    InSituExperiment
```

Read the `InSituExperiment` object with:

```{eval-rst}
.. module:: insitupy.experiment
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated

    InSituExperiment.read
```

## Core Data Classes

Data classes are used to store the different modalities.

### Cellular data

```{eval-rst}
.. module:: insitupy.dataclasses
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated

    dataclasses.CellData
    dataclasses.MultiCellData
    dataclasses.BoundariesData
```

### Image data

```{eval-rst}
.. module:: insitupy.dataclasses
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated

    dataclasses.ImageData
```

### Geometric data

```{eval-rst}
.. module:: insitupy.dataclasses
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated

    dataclasses.ShapesData
    dataclasses.AnnotationsData
    dataclasses.RegionsData
```

The different data classes can be read using following functions:

```{eval-rst}
.. module:: insitupy.dataclasses
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated

    dataclasses.read_celldata
    dataclasses.read_multicelldata
    dataclasses.read_shapesdata
```

## Read external data

### Individual datasets

```{eval-rst}
.. module:: insitupy._core
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated

    read_qupath
    read_xenium
```

### Multiple datasets

```{eval-rst}
.. module:: insitupy.experiment
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated

    read_qupath_project
```


## Plotting

```{eval-rst}
.. module:: insitupy.plotting
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated

    plotting.plot_spatial
    plotting.plot_cellular_composition
    plotting.cell_abundance_along_axis
    plotting.cell_expression_along_axis
    plotting.plot_volcano
    plotting.plot_colorlegend
    plotting.plot_overview
```

