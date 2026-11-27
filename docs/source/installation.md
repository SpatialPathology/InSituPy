# Installation

Make sure you have Conda installed on your system before proceeding with these steps. If not, you can install Miniconda or Anaconda from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html).

## Create and activate a conda environment

   ```bash
   conda create --name insitupy python=3.13
   conda activate insitupy
   ```

## Install Jupyter kernel

To ensure that the InSituPy package is available as a kernel in Jupyter notebooks within your conda environment, you can follow the instructions [here](https://ipython.readthedocs.io/en/stable/install/kernel_install.html).

## Install InSituPy

### Method 1: Install from PyPi

   ```bash
   pip install insitupy-spatial
   ```

This base installation does not include the interactive viewer (napari, Qt) and is suitable for headless and HPC environments (the `spatialdata` extra brings napari back, see below).

### Optional: Install the interactive viewer

To use `.show()` (the napari-based viewer), install the `viewer` extra:

   ```bash
   pip install "insitupy-spatial[viewer]"
   ```

InSituPy currently requires `zarr>=3.0.0` and targets the zarr v3 format. Legacy zarr v2 workflows are only partially supported and not tested.

### Optional: Install with SpatialData support (`spatialdata>=0.8.0,<0.9.0`)

   ```bash
   pip install insitupy-spatial[spatialdata]
   ```

Note: the `spatialdata` extra installs `spatialdata[extra]`, which pulls in napari (via `napari-spatialdata`). An environment with this extra therefore is not free of the GUI stack. For the InSituPy viewer, still install the `viewer` extra.

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
   pip install git+https://github.com/SpatialPathology/InSituPy.git
   ```

## Upgrading from 0.12

napari is no longer installed with InSituPy by default. Existing environments keep their napari
until they are recreated, but new environments need `pip install "insitupy-spatial[viewer]"` for
`.show()`. Without it, `.show()` raises an `ImportError` that names the extra. For development,
`pip install -e ".[dev]"` includes the viewer.

InSituPy now depends on `opencv-python-headless` instead of `opencv-python`. In an existing
environment, run `pip uninstall -y opencv-python opencv-python-headless` and then
`pip install opencv-python-headless` (or reinstall InSituPy). Having both wheels installed puts
two copies of the `cv2` module in the same folder, and uninstalling either one later breaks the
other.

If you rely on `cv2.imshow` or other OpenCV GUI functions in your own code, install
`opencv-python` yourself (after removing the headless wheel).
