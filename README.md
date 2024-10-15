# InSituPy: Python Package for the Analysis of _Xenium In Situ_ Data

<p align="center">
   <img src="logo/insitupy_logo.png" width="500">
</p>

InSituPy is a Python package designed to facilitate the analysis of in situ sequencing data generated with the [_Xenium In Situ_](https://www.10xgenomics.com/platforms/xenium) methodology. With InSituPy, you can easily load, visualize, and analyze the data, enabling and simplifying the comprehensive exploration of spatial gene expression patterns within tissue sections.

## Installation

To install InSituPy within a conda environment, you can follow these steps:

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/jwrth/InSituPy.git
   ```

2. Navigate to the cloned repository and select the right branch. Since the repository is under very active development it can make sense to use the `dev` branch including also the most recent developments:

   ```bash
   cd InSituPy

   # Optionally: switch to dev branch
   git checkout dev
   ```

3. Create and activate a conda environment:

   ```bash
   conda create --name insitupy python=3.9
   conda activate insitupy
   ```

4. **(Windows Only)** Download and install the `annoy` package from the provided wheel file. If your Python version is different from `3.9`, make sure to download the correct wheel [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#annoy) and adapt the installation code accordingly:

   ```bash
   pip install wheels\annoy-1.17.0-cp39-cp39-win_amd64.whl
   ```

   *(Note: Skip this step if you're on Mac or Linux.)*

5. Install the rest of the required packages using `pip` within the conda environment:

   ```bash
   # for installation without napari use
   pip install .

   # for installation with napari use
   pip install .[napari]

   # for developmental purposes add the -e flag
   pip install -e .[napari]
   ```

Make sure you have Conda installed on your system before proceeding with these steps. If not, you can install Miniconda or Anaconda from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html).

To ensure that the InSituPy package is available as a kernel in Jupyter notebooks within your conda environment, you can follow the instructions [here](https://ipython.readthedocs.io/en/stable/install/kernel_install.html).

## Getting Started

For detailed instructions on using InSituPy, refer to the [official documentation](https://InSituPy.readthedocs.io). The documentation provides comprehensive guides on installation, usage, and advanced features.

## Tutorials

Explore the tutorials in `./notebooks/` to learn how to use InSituPy:

1. [Download example data for tutorial](notebooks/01_InSituPy_demo_download_data.ipynb) - Download _Xenium In Situ_ example data for the subsequent tutorials.
2. [Registration of additional images](notebooks/02_InSituPy_demo_register_images.ipynb) - Learn how to register additional images.
3. [Basic analysis functionalities](notebooks/03_InSituPy_demo_analyze.ipynb) - Learn about the basic functionalities, such as loading of data, basic preprocessing and interactive visualization with napari.
4. [Add annotations](notebooks/04_InSituPy_demo_annotations.ipynb) - Learn how to add annotations from external software such as [QuPath](https://qupath.github.io/).

## Features

- **Data Preprocessing:** InSituPy provides functions for normalizing, filtering, and transforming raw in situ transcriptomics data.

- **Interactive Visualization:** Create interactive plots using [napari](https://napari.org/stable/#) to easily explore spatial gene expression patterns.

- **Annotation:** Annotate _Xenium In Situ_ data in the napari viewer or import annotations from external tools like [QuPath](https://qupath.github.io/).

## Contributing

Contributions are welcome! If you find any issues or have suggestions for new features, please open an [issue](https://github.com/jwrth/InSituPy/issues) or submit a pull request.

## License

InSituPy is licensed under the [BSD-3-Clause](LICENSE).

---

**InSituPy** is developed and maintained by [jwrth](https://github.com/jwrth). Feedback is highly appreciated and hopefully **InSituPy** helps you with your _Xenium In Situ_ analysis. The package is thought to be a starting point to simplify the analysis of in situ sequencing data in Python and it would be exciting to integrate functionalities into larger and more comprehensive data structures.
