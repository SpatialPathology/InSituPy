# Tutorials

The tutorials show how to work with **InSituPy**'s two core classes: `InSituData` for a single
tissue section and `InSituExperiment` for a study with many samples. New to InSituPy? Read
[InSituPy at a glance](../overview.md) first for how the data is organised.

## Learn the basics

Start here. Download the demo data and work through the single-sample tutorials in order, then
learn how to bring in your own data and how to look at it.

```{eval-rst}
.. card:: Single-sample analysis
    :link: 01_demo_analysis/index
    :link-type: doc
    :link-alt: Analysis demo

    Step-by-step demonstration on how to perform data analysis using `InSituPy` and the `InSituData` class.

.. card:: Data import
    :link: 03_data_import/index
    :link-type: doc
    :link-alt: Data import tutorials

    Tutorials explaining how to import data from different technologies or tools.

.. card:: Plotting functionalities
    :link: 04_plotting/index
    :link-type: doc
    :link-alt: Plotting tutorials

    Tutorials introducing different plotting functionalities.

```

## Go further

Work with many samples, save and reload your data, and exchange data with other tools.

```{eval-rst}
.. card:: Multi-sample analysis
    :link: 02_multisample_analysis/index
    :link-type: doc
    :link-alt: Multi-sample analysis tutorials

    This set of tutorials focuses on the analysis of multiple samples using the `InSituExperiment` class.

.. card:: Data I/O - Saving and Loading
    :link: 00_io/index
    :link-type: doc
    :link-alt: Data I/O tutorials

    How to save and reload ``InSituData`` and ``InSituExperiment`` objects - full saves, modality-specific partial saves, and reloading individual components.

.. card:: SpatialData Integration
    :link: 07_spatialdata/index
    :link-type: doc
    :link-alt: SpatialData conversion tutorials

    Tutorials for converting between InSituPy and SpatialData formats for integration with the scverse ecosystem.

```

## Reference and manuscript

```{eval-rst}
.. card:: Manuscript-related analyses
    :link: 05_publication/index
    :link-type: doc
    :link-alt: Publication-related analyses

    Notebooks including analyses shown in the manuscript.

.. card:: Benchmarkings
    :link: 06_benchmarkings/index
    :link-type: doc
    :link-alt: Benchmarking notebooks

    Performance measurements and comparisons against alternative readers - useful when deciding how to read large datasets or quantify signal on very large images.

```


```{toctree}
:hidden: false
:maxdepth: 2

01_demo_analysis/index.md
03_data_import/index.md
04_plotting/index.md
02_multisample_analysis/index.md
00_io/index.md
07_spatialdata/index.md
05_publication/index.md
06_benchmarkings/index.md
```
