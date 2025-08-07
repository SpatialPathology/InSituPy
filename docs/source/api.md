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

    AnnotationsData
    BoundariesData
    CellData
    ImageData
    MultiCellData
    RegionsData
```

### Reading Data Classes

```{eval-rst}
.. module:: insitupy.dataclasses
.. currentmodule:: insitupy

.. autosummary::
    :toctree: generated

    read_celldata
    read_multicelldata
    read_shapesdata
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

