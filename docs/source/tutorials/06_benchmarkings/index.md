# Benchmarkings

Notebooks measuring **InSituPy**'s performance and comparing its readers against alternatives.
These are reference measurements rather than step-by-step tutorials - useful when deciding how to
read large datasets or how to quantify signal on images that do not fit in memory.

```{eval-rst}
.. card:: Performance tests
    :link: InSituPy_performance_tests
    :link-type: doc

    Runtime and memory measurements across the core ``InSituPy`` operations.

.. card:: Memory-efficient fluorescence quantification
    :link: InSituPy_quantify_fluorescence_benchmarking
    :link-type: doc

    Benchmarks the tiled implementation of ``quantify_signal`` against the direct one, for
    fluorescence images too large to load into memory at once.

.. card:: ``read_xenium`` vs. ``spatialdata-io``
    :link: InSituPy_read_functions_spatialdata
    :link-type: doc

    Compares InSituPy's native Xenium reader against the ``spatialdata-io`` reading function.

.. card:: Reading Vannan et al. data with SpatialData
    :link: InSituPy_Vannan_paper_import_spatialdata
    :link-type: doc

    Reads the multi-sample dataset from `Vannan et al. <https://www.nature.com/articles/s41588-025-02080-x>`_
    via SpatialData, as a counterpart to the InSituPy-native import tutorial.
```

```{toctree}
:hidden: false
:maxdepth: 1

InSituPy_performance_tests
InSituPy_quantify_fluorescence_benchmarking
InSituPy_read_functions_spatialdata
InSituPy_Vannan_paper_import_spatialdata
```
