import os
from datetime import datetime
from numbers import Number
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from insitupy.io.plots import save_and_show_figure
from insitupy.utils.utils import get_nrows_maxcols

from ..io.files import read_json


def find_xenium_outputs(
    path: Union[str, os.PathLike, Path],
    startswith: str = 'output-XET'
    ) -> List:
    print(f"Searching for directories starting with '{startswith}' in {str(path)}")
    search_results = []
    for root, dirs, files in os.walk(path):
        root = Path(root)
        for d in dirs:
            if d.startswith(startswith):
                p = root / d
                search_results.append(p)

    print(f"Found {len(search_results)} Xenium output directories.")
    return search_results

def collect_qc_data(
    data_folders: List[Union[str, os.PathLike, Path]]
    ) -> pd.DataFrame:

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

