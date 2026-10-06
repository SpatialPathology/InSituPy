# Welcome to InSituPy's documentation!

```{image} _static/img/insitupy_logo_with_name_wo_bg.png
:alt: InSituPy logo
:class: dark-light p-2
:width: 500px
:align: center
```

**InSituPy** is a Python package for the analysis of single-cell spatial transcriptomics data. With InSituPy, you can read, visualize, and analyze the spatially resolved gene expression within one dataset but also across different datasets. Further, it provides a general structure for organizing multiple datasets and its corresponding metadata.

Currently the analysis is focused on data from the [_Xenium In Situ_](https://www.10xgenomics.com/platforms/xenium) methodology. Readers for [Visium](https://www.10xgenomics.com/platforms/visium) and [QuPath](https://qupath.github.io/) data are available as well, as is conversion to and from [SpatialData](https://spatialdata.scverse.org).

```{eval-rst}
.. note::
   !!!Warning: This repository is under very active development and it cannot be ruled out that changes might impair backwards compatibility. If you observe any such thing, please feel free to contact us to solve the problem. Thanks!
```

## Features

A key feature of InSituPy is its **hierarchical data structure**: an `InSituExperiment` holds many samples, each an `InSituData` that keeps images, cells, transcripts, annotations and more together. See [InSituPy at a glance](overview.md) for how it works.

```{image} _static/img/insitupy_data_hierarchy.svg
:alt: The InSituPy data hierarchy
:width: 800px
:align: center
```

Additional features include:
- **Data Preprocessing:** InSituPy provides functions for normalizing, filtering, and transforming raw in situ transcriptomics data.
- **Interactive Visualization:** Create interactive plots using [napari](https://napari.org/stable/#) to easily explore spatial gene expression patterns.
- **Annotation:** Annotate _Xenium In Situ_ data in the napari viewer or import annotations from external tools like [QuPath](https://qupath.github.io/).
- **Multi-sample analysis:** Perform analysis on an experiment-level, i.e. with multiple samples at once.

## Getting started

```{eval-rst}
.. card:: InSituPy at a glance
    :link: overview
    :link-type: doc
    :link-alt: InSituPy at a glance

    New here? What **InSituPy** does and how it organises your data, in plain words.

.. card:: Installation
    :link: installation
    :link-type: doc
    :link-alt: Installation

    Learn how to install **InSituPy**.

.. card:: Tutorials
    :link: tutorials/index
    :link-type: doc
    :link-alt: Tutorials

    Tutorials to help you get started with **InSituPy**.

.. card:: API
    :link: api
    :link-type: doc
    :link-alt: API

    Application Programming Interface.

```

## Contributing

Contributions are welcome! If you find any issues or have suggestions for new features, please open an [issue](https://github.com/SpatialPathology/InSituPy/issues) or submit a pull request. To engage in discussions and start working collectively on InSituPy, feel free to post in our [Zulip chat](https://insitupy.zulipchat.com).

```{toctree}
:hidden: false
:maxdepth: 3
:glob:

overview.md
installation.md
ai_integration.md
tutorials/*
api.md
```