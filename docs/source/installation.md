# Installation

## Prerequisites

**Create and activate a conda environment:**

   ```bash
   conda create --name insitupy python=3.10
   conda activate insitupy
   ```

## Method 1: From PyPi

   ```bash
   pip install insitupy-spatial
   ```

## Method 2: Installation from Cloned Repository

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

## Method 3: Direct Installation from GitHub

1. **Install directly from GitHub:**

   ```bash
   # for installation without napari use
   pip install git+https://github.com/SpatialPathology/InSituPy.git
   ```

Make sure you have Conda installed on your system before proceeding with these steps. If not, you can install Miniconda or Anaconda from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html).

To ensure that the InSituPy package is available as a kernel in Jupyter notebooks within your conda environment, you can follow the instructions [here](https://ipython.readthedocs.io/en/stable/install/kernel_install.html).
