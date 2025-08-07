# API

```{eval-rst}

Import InSituPy as::

    import insitupy

.. module:: insitupy
```

## Core Data Objects

For individual datasets:

```{eval-rst}
.. module:: insitupy._core
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated

    InSituData
    InSituData.read
```

For handling multiple samples:

```{eval-rst}
.. module:: insitupy.experiment
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated

    InSituExperiment
    InSituExperiment.read
```

## Reading external data

```{eval-rst}
.. module:: insitupy._core
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated

    read_qupath
    read_xenium
```

## Core Data Classes

```{eval-rst}
.. module:: insitupy.dataclasses
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated

    dataclasses.AnnotationsData
    dataclasses.BoundariesData
    dataclasses.CellData
    dataclasses.ImageData
    dataclasses.MultiCellData
    dataclasses.RegionsData
```

### Reading Data Classes

```{eval-rst}
.. module:: insitupy.dataclasses
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated

    dataclasses.read_celldata
    dataclasses.read_multicelldata
    dataclasses.read_shapesdata
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

