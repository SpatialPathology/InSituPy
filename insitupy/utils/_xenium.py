import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from numbers import Number
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from insitupy.plotting.save import save_and_show_figure
from insitupy.utils.utils import get_nrows_maxcols

logger = logging.getLogger(__name__)

from .._io.files import read_json


def find_xenium_outputs(
    path: Union[str, os.PathLike, Path],
    startswith: str = 'output-XET',
    max_depth: int = None,
    threads: int = 4,
) -> List[Path]:
    """
    Search for Xenium output directories more efficiently by:
    - Using iterdir() instead of os.walk to avoid unnecessary recursion
    - Pruning subtrees that cannot contain matches
    - Optionally parallelizing directory scanning with threads
    """
    path = Path(path)
    logger.info(f"Searching for directories starting with '{startswith}' in {path}")

    results = []

    def _scan(directory: Path, depth: int) -> List[Path]:
        found = []
        try:
            entries = list(directory.iterdir())
        except PermissionError:
            return found

        subdirs = []
        for entry in entries:
            if not entry.is_dir():
                continue
            if entry.name.startswith(startswith):
                found.append(entry)
                # Xenium output dirs don't nest - no need to recurse into them
            elif max_depth is None or depth < max_depth:
                subdirs.append(entry)

        if subdirs:
            with ThreadPoolExecutor(max_workers=threads) as executor:
                futures = {executor.submit(_scan, d, depth + 1): d for d in subdirs}
                for future in as_completed(futures):
                    found.extend(future.result())

        return found

    results = _scan(path, depth=0)
    logger.info(f"Found {len(results)} Xenium output directories.")
    return results

def collect_qc_data(
    data_folders: List[Union[str, os.PathLike, Path]]
    ) -> pd.DataFrame:
    """Collect QC metadata from a list of Xenium output directories.

    Parses the run date from the folder name and reads a fixed set of QC
    fields (cell count, transcripts per cell, etc.) from each
    ``experiment.xenium`` metadata file.

    Args:
        data_folders: Paths to Xenium output directories.

    Returns:
        A :class:`~pandas.DataFrame` with one row per directory and columns
        for date, run name, slide ID, region, preservation method, and key
        QC metrics.
    """
    cats = ["date", "run_name", "slide_id", "region_name", "preservation_method",
            "num_cells", "transcripts_per_cell",
            "transcripts_per_100um", "panel_organism", "panel_tissue_type"]

    results = []
    for f in data_folders:
        date_string = Path(f).stem.split("__")[-2]
        date_object = datetime.strptime(date_string, "%Y%m%d")
        metadata = read_json(Path(f) / "experiment.xenium")
        extracted = [date_object] + [metadata[c] for c in cats[1:]]
        results.append(extracted)

    data = pd.DataFrame(results, columns=cats)
    return data


def plot_qc(
    data: pd.DataFrame,
    x: str = "preservation_method",
    cats: List[str] = ["num_cells", "transcripts_per_cell", "transcripts_per_100um"],
    max_cols: int = 4,
    fontsize: int = 22,
    size: Number = 10,
    savepath: Union[str, os.PathLike, Path] = None,
    save_only: bool = False,
    dpi_save: int = 300
    ):
    """Plot Xenium QC metrics as a strip-plot grid, grouped by a categorical column.

    Args:
        data: DataFrame returned by :func:`collect_qc_data`.
        x: Column used for the x-axis grouping.
        cats: Numeric columns to plot — one subplot per entry.
        max_cols: Maximum number of columns in the subplot grid.
        fontsize: Base font size applied globally via ``rcParams``.
        size: Marker size for the strip plot.
        savepath: If provided, save the figure to this path.
        save_only: If True, close the figure after saving without displaying it.
        dpi_save: Resolution used when saving.
    """
    # set plotting parameters
    plt.rcParams.update({
    'font.size': fontsize,          # Base font size
    'axes.titlesize': fontsize,     # Title font size
    'axes.labelsize': fontsize,     # Axis label font size
    'xtick.labelsize': fontsize,    # X-tick label font size
    'ytick.labelsize': fontsize,    # Y-tick label font size
    'legend.fontsize': fontsize,    # Legend font size
    'figure.titlesize': fontsize    # Figure title font size
})

    # plot
    n_plots, nrows, ncols = get_nrows_maxcols(len(cats), max_cols=max_cols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(9*ncols, 8*nrows))

    if n_plots > 1:
        axs = axs.ravel()
    else:
        axs = [axs]

    for i, cat in enumerate(cats):
        sns.boxplot(data=data, x=x, y=cat,
                    #color="w",
                    hue="panel_tissue_type",
                    #boxprops={"facecolor": 'w'}, fliersize=0,
                    ax=axs[i],
                    )
        # sns.stripplot(data=data,
        #               x=x, y=cat,
        #               hue="panel_tissue_type",
        #               size=size,
        #               ax=axs[i]
        #               )
        axs[i].set_title(cat)
        axs[i].set_ylabel(None)

        if i+1 == ncols:
            # move legend out of the plot
            axs[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            # remove legend
            axs[i].get_legend().remove()

    plt.show()

    save_and_show_figure(savepath=savepath, fig=fig, save_only=save_only, dpi_save=dpi_save, tight=True)


def copy_files_from_xenium_output(
    source_dir,
    target_dir,
    filename,
    xenium_filename: str = "experiment.xenium"
    ):
    """
    Copies specified files from subdirectories within a source directory to a target directory,
    renaming them based on metadata found in a signature file.

    Args:
        source_dir (str): The path to the source directory containing subdirectories.
        target_dir (str): The path to the target directory where files will be copied.
        filename (str): The name of the file to be copied from each subdirectory.
        signature_filename (str, optional): The name of the signature file used to identify
                                            valid subdirectories and extract metadata. Defaults to "experiment.xenium".

    Raises:
        FileNotFoundError: If the specified file or signature file does not exist in a subdirectory.

    Example:
        copy_files_from_xenium_output("/path/to/source", "/path/to/target", "data.txt")
    """
    # Ensure the target directory exists
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    # Iterate through all folders in the source directory
    for folder in Path(source_dir).glob('*'):
        if folder.is_dir():
            # check if it is a Xenium output directory
            xenium_file = folder / xenium_filename
            if xenium_file.exists():
                logger.info(f"Found Xenium output directory: {folder}")
                # Check if the specified file exists in the current folder
                file_path = folder / filename

                # get metadata
                slide_id = read_json(xenium_file)["slide_id"]
                region_name = read_json(xenium_file)["region_name"]
                if file_path.exists():
                    # Copy the file to the target directory
                    shutil.copy(file_path, target_path / f"{slide_id}__{region_name}__{filename}")
                    logger.info(f"\tCopied {file_path} to {target_path}")
                else:
                    logger.warning("File not found in directory.")

