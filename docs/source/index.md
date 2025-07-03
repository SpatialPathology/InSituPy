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
   This repository is under very active development and it cannot be guaranteed that releases contain changes that might impair backwards compatibility. If you observe any such thing, please feel free to contact us to solve the problem. Thanks!
```

## Features

- **Data Preprocessing:** InSituPy provides functions for normalizing, filtering, and transforming raw in situ transcriptomics data.
- **Interactive Visualization:** Create interactive plots using [napari](https://napari.org/stable/#) to easily explore spatial gene expression patterns.
- **Annotation:** Annotate _Xenium In Situ_ data in the napari viewer or import annotations from external tools like [QuPath](https://qupath.github.io/).
- **Multi-sample analysis:** Perform analysis on an experiment-level, i.e. with multiple samples at once.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for new features, please open an [issue](https://github.com/SpatialPathology/InSituPy/issues) or submit a pull request.

```{toctree}
:hidden: false
:maxdepth: 3
:glob:

installation.md
tutorials/*
api.md
```