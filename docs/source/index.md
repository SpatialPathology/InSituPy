# Welcome to InSituPy's documentation!

```{image} _static/img/insitupy_logo_with_name_wo_bg.png
:alt: InSituPy logo
:class: dark-light p-2
:width: 500px
:align: center
```

**InSituPy** is a Python package for the analysis of single-cell spatial transcriptomics data. With InSituPy, you can read, visualize, and analyze the spatially resolved gene expression within one dataset but also across different datasets. Further, it provides a general structure for organizing multiple datasets and its corresponding metadata.

Currently the analysis is focused on data from the [_Xenium In Situ_](https://www.10xgenomics.com/platforms/xenium) methodology but a broader range of reading functions will be implemented in the future.

```{eval-rst}
.. note::
   !!!Warning: This repository is under very active development and it cannot be ruled out that changes might impair backwards compatibility. If you observe any such thing, please feel free to contact us to solve the problem. Thanks!
```

## Features

A key feature of InSituPy is its **hierarchical data structure**, centered around the `InSituExperiment` and `InSituData` objects:
- `InSituData`: Represents and manages at the individual sample level. It integrates all modalities of spatial omics datasets, including cellular readouts, cellular boundaries, images, transcripts, regions, and annotations.
- `InSituExperiment`: Aggregates multiple `InSituData` instances and links them with associated metadata, enabling cross-sample analysis and organization.

<p align="center">
   <img src="https://github.com/SpatialPathology/InSituPy/blob/main/docs/source/_static/img/insitupy_data_structure.svg?raw=true" width="800">
</p>

Additional features include:
- **Data Preprocessing:** InSituPy provides functions for normalizing, filtering, and transforming raw in situ transcriptomics data.
- **Interactive Visualization:** Create interactive plots using [napari](https://napari.org/stable/#) to easily explore spatial gene expression patterns.
- **Annotation:** Annotate _Xenium In Situ_ data in the napari viewer or import annotations from external tools like [QuPath](https://qupath.github.io/).
- **Multi-sample analysis:** Perform analysis on an experiment-level, i.e. with multiple samples at once.

## Getting started

```{eval-rst}
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

installation.md
tutorials/*
api.md
```