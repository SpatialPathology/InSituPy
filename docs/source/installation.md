# Installation

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