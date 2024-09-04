import os
from numbers import Number
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..io.files import read_json
from ..io.plots import save_and_show_figure
from .utils import get_nrows_maxcols


def find_xenium_outputs(
    path: Union[str, os.PathLike, Path],
    startswith: str = 'output-XET'
    ) -> List:
    print(f"Searching for directories starting with '{startswith}' in {str(path)}")
    search_results = []
    for root, dirs, files in os.walk(path):
        for d in dirs:
            if d.startswith(startswith):
                search_results.append(os.path.join(root, d))

    print(f"Found {len(search_results)} Xenium output directories.")
    return search_results

def collect_qc_data(
    data_folders: List[Union[str, os.PathLike, Path]]
    ) -> pd.DataFrame:

    cats = ["run_name", "region_name", "preservation_method",
            "num_cells", "transcripts_per_cell",
            "transcripts_per_100um", "panel_organism", "panel_tissue_type"]

    results = []
    for f in data_folders:
        metadata = read_json(Path(f) / "experiment.xenium")
        extracted = [metadata[c] for c in cats]
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
    # plot
    plt.rcParams.update({'font.size': fontsize})
    n_plots, nrows, ncols = get_nrows_maxcols(cats, max_cols=max_cols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(9*ncols, 8*nrows))

    if n_plots > 1:
        axs = axs.ravel()
    else:
        axs = [axs]

    for i, cat in enumerate(cats):
        sns.boxplot(data=data, x=x, y=cat, color="w",
                    boxprops={"facecolor": 'w'}, fliersize=0,
                    ax=axs[i], )
        sns.stripplot(data=data,
                      x=x, y=cat,
                      hue="panel_tissue_type",
                      size=size,
                      ax=axs[i]
                      )
        axs[i].set_title(cat)
        axs[i].set_ylabel(None)

        if i+1 == ncols:
            # move legend out of the plot
            axs[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            # remove legend
            axs[i].get_legend().remove()

    save_and_show_figure(savepath=savepath, fig=fig, save_only=save_only, dpi_save=dpi_save, tight=True)
