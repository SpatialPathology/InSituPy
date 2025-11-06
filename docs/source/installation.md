# Installation

Make sure you have Conda installed on your system before proceeding with these steps. If not, you can install Miniconda or Anaconda from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html).

## Create and activate a conda environment

   ```bash
   conda create --name insitupy python=3.10
   conda activate insitupy
   ```

## Install Jupyter kernel

To ensure that the InSituPy package is available as a kernel in Jupyter notebooks within your conda environment, you can follow the instructions [here](https://ipython.readthedocs.io/en/stable/install/kernel_install.html).

## Install InSituPy

### Method 1: Install from PyPi

   ```bash
   pip install insitupy-spatial
   ```

#### Optional: Install with GUI support (napari viewer)

If you want to use the graphical interface features powered by [napari](https://napari.org/dev/index.html), install with the gui extra:

   ```bash
   pip install insitupy-spatial[gui]
   ```

### Method 2: Installation from Cloned Repository

1. **Clone the repository to your local machine:**

   ```bash
   git clone https://github.com/SpatialPathology/InSituPy.git
   ```

2. **Navigate to the cloned repository and select the right branch:**

   ```bash
   cd InSituPy

   # Optionally: switch to dev branch
   git checkout dev
   ```

3. **Install the required packages using `pip` within the conda environment:**

   ```bash
   # basic installation
   pip install .

   # for developmental purposes add the -e flag
   pip install -e .
   ```

### Method 3: Direct Installation from GitHub

1. **Install directly from GitHub:**

   ```bash
   # for installation without napari use
   pip install git+https://github.com/SpatialPathology/InSituPy.git
   ```
