# InSituPy: A framework for histology-guided, multi-sample  analysis of single-cell spatial transcriptomics data

<p align="center">
   <img src="https://github.com/SpatialPathology/InSituPy/blob/main/docs/source/_static/img/insitupy_logo_with_name_wo_bg.png?raw=true" width="500">
</p>

InSituPy is a Python package designed to facilitate the analysis of single-cell spatial transcriptomics data. With InSituPy, you can easily load, visualize, and analyze the data, enabling and simplifying the comprehensive exploration of spatial gene expression patterns within tissue sections and across multiple samples.
Currently the analysis is focused on data from the [_Xenium In Situ_](https://www.10xgenomics.com/platforms/xenium) methodology but a broader range of reading functions will be implemented in the future.

## Latest changes

*!!!Warning: This repository is under very active development and it cannot be ruled out that changes might impair backwards compatibility. If you observe any such thing, please feel free to contact us to solve the problem. Thanks!*

For the latest developments check out the [releases](https://github.com/SpatialPathology/InSituPy/releases).

### Comming soon: Interface to SpatialData

A interface to convert data from and to [SpatialData](https://github.com/scverse/spatialdata) is currently implemented and will be released soon. If you are interested in this, feel free to contact us to get early access.

### Big improvements in `0.9.0`

This version contains multiple bigger changes:
- Implementation of `generate_pseudobulk` function
- Working with multiple open napari viewers in parallel
- Sync and save buttons in napari viewer
- Implementation of readers for multiplex-IF data exported from QuPath. For scripts to work with QuPath and InSituPy see [this repository](https://github.com/SpatialPathology/InSituPy-QuPath)
- Changed syntax of the `plotting` submodule: The prefix `"plot_"` was removed from all functions. E.g. `plot_spatial` can be now called with `.pl.spatial`.
- Completely revised code in `.pl.spatial` introducing different configuration objects to format the plots.
- Improved documentation

## Getting started

### Overall data structure

A key feature of InSituPy is its hierarchical data structure, centered around the `InSituExperiment` and `InSituData` objects:
- `InSituData`: Represents and manages at the individual sample level. It integrates all modalities of spatial omics datasets, including cellular readouts, cellular boundaries, images, transcripts, regions, and annotations.
- `InSituExperiment`: Aggregates multiple `InSituData` instances and links them with associated metadata, enabling cross-sample analysis and organization.

<p align="center">
   <img src="https://github.com/SpatialPathology/InSituPy/blob/main/docs/source/_static/img/insitupy_data_structure.svg?raw=true" width="800">
</p>


### Documentation

For detailed instructions on using InSituPy, refer to the [official documentation](https://InSituPy.readthedocs.io).

InSituPy works best within *Jupyter Lab* or *Jupyter Notebook* sessions. If you are not familiar with these platforms, see the documentation of [Project Jupyter](https://jupyter.org/).

## Installation

Make sure you have Conda installed on your system before proceeding with these steps. If not, you can install Miniconda or Anaconda from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html).

**Create and activate a conda environment:**

   ```bash
   conda create --name insitupy python=3.10
   conda activate insitupy
   ```

**Install from PyPi:**

   ```bash
   pip install insitupy-spatial
   ```

## Optional: Install with GUI support (napari viewer):

If you want to use the graphical interface features powered by [napari](https://napari.org/dev/index.html), install with the gui extra:

   ```bash
   pip install insitupy-spatial[gui]
   ```

To ensure that the InSituPy package is available as a kernel in Jupyter notebooks within your conda environment, you can follow the instructions [here](https://ipython.readthedocs.io/en/stable/install/kernel_install.html).

For alternative installation strategies see the [documentation](https://insitupy.readthedocs.io/en/latest/installation.html).


## Features

- **Data storage**: Store data on both the single sample level and the multi-sample level using the `InSituData` and `InSituExperiment` objects.
- **Data Preprocessing:** InSituPy provides functions for normalizing, filtering, and transforming raw in situ transcriptomics data.
- **Interactive Visualization:** Create interactive plots using [napari](https://napari.org/stable/#) to easily explore spatial gene expression patterns.
- **Annotation:** Annotate _Xenium In Situ_ data in the napari viewer or import annotations from external tools like [QuPath](https://qupath.github.io/).
- **Multi-sample analysis:** Perform analysis on an experiment-level, i.e. with multiple samples at once.

## QuPath

We try to develop InSituPy alongside the Bioimage Analysis tool [QuPath](https://qupath.github.io). QuPath has great functionalities to visualize whole slide image data, add annotations, generate segmentations or analyze signal intensities. Scripts to simplify the connection between QuPath and InSituPy, we collect [here](https://github.com/SpatialPathology/InSituPy-QuPath). This includes:
- Export of annotations as GEOJSON from QuPath
- Export of images as OME-TIFF from QuPath
- Collected export of data from a multiplexed IF image to be imported into InSituPy. Import can be performed using either `read_qupath` or `read_qupath_project`. For cell and nucleus segmentation of multiplexed IF images we recommend using [Instanseg](https://github.com/instanseg/instanseg).

## Contributing

Contributions are welcome! If you find any issues or have suggestions for new features, please open an [issue](https://github.com/SpatialPathology/InSituPy/issues) or submit a pull request.

## Citation

If you use `InSituPy` in your work, please cite the [preprint](https://www.biorxiv.org/content/10.1101/2025.03.07.641860v1) as follows:

> InSituPy – A Framework for Histology-Guided, Multi-Sample Analysis of Single-Cell Spatial Transcriptomics Data. <br>Wirth, Johannes, Anna Chernysheva, Birthe Lemke, Isabel Giray, Aitana Egea Lavandera, and Katja Steiger.<br>
bioRxiv, March 12, 2025. https://doi.org/10.1101/2025.03.07.641860.

## License

InSituPy is licensed under the [BSD-3-Clause](LICENSE).

---

**InSituPy** is developed and maintained by [Johannes Wirth](https://github.com/jwrth) and [Anna Chernysheva](https://github.com/annachernysheva179). Feedback is highly appreciated and hopefully **InSituPy** helps you with your analysis of spatial transcriptomics data. The package is thought to be a starting point to simplify the analysis of in situ sequencing data in Python and it would be exciting to integrate functionalities for larger and more comprehensive data structures. Currently, the framework focuses on the analysis of _Xenium In Situ_ data but it is planned to integrate more methodologies and any support on this is highly welcomed.
