import os
import warnings
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import scanpy as sc
from anndata import AnnData
from matplotlib.axes._axes import Axes

from insitupy.dataclasses._utils import _get_cell_layer
from insitupy.experiment.data import InSituExperiment
from insitupy.plotting.save import save_and_show_figure
from insitupy.utils._checks import _calculate_single_metrics


def _calculate_max_cell_widths_and_sum(df, multiplier=0.13):
    """
    Calculate the maximum cell width for each column based on text length, including the column name in the calculation, and return the sum of them.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        multiplier (int): The multiplier to adjust the width based on text length.

    Returns:
        dict: A dictionary with column names as keys and maximum widths as values.
        int: The sum of the maximum widths.
    """
    max_widths = {}
    total_width = 0
    for col in df.columns:
        # Calculate the maximum width for each column based on the length of the text in the cells and the column name
        max_width = max(df[col].apply(lambda x: len(str(x)) * multiplier).max(), len(col) * multiplier)
        max_widths[col] = max_width
        total_width += max_width
    return max_widths, total_width

def _custom_bar(ax: Axes, val: float, max: float, color: str = None, rect_kw: dict = {}, fontsize: int = 10):
        """
        Custom function to create a horizontal bar plot.

        Args:
            ax (Axes): The axes on which to plot.
            val (float): The value to plot.
            max (float): The maximum value for the x-axis.
            color (str, optional): The color of the bar.
            rect_kw (dict, optional): Additional keyword arguments for the rectangle.
            fontsize (int, optional): Font size for the value labels. Defaults to 10.

        Returns:
            bar: The bar plot.
        """
        # Create a horizontal bar plot with the specified value and maximum
        bar = ax.barh(y=0.5, left=1, width=val, height=0.8, fc=color, ec="None", zorder=0.05)
        ax.set_xlim(0, max + 10)
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(['{:.0f}'.format(x) for x in ax.get_xticks()])
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        for r in bar:
            r.set(**rect_kw)
        for rect in bar:
            width = rect.get_width()
            ax.text(width + 1, rect.get_y() + rect.get_height() / 2, f'{width:.0f}',
                    ha='left', va='center', fontsize=fontsize)
        return bar


def overview(
    data: InSituExperiment,
    cells_layer: Optional[str] = None,
    columns_to_plot: List[str] = [],
    layer: str = None,
    force_layer: bool = False,
    index: bool = True,
    qc_width: float = 3.0,
    fontsize: int = 10,
    savepath: Union[str, os.PathLike, Path] = None,
    save_only: bool = False,
    dpi_save: int = 300
    ):
    """
    Plots an overview table with metadata and quality control metrics.

    Args:
        data (InSituExperiment): The experiment object containing datasets and metadata.
        cells_layer (str, optional): The layer of cells to use. Defaults to None.
        columns_to_plot (List[str]): List of column names to include in the plot.
        layer (str, optional): The layer of the AnnData object to use for calculations. If None, the function will use the main matrix (adata.X) or the 'counts' layer if the main matrix does not contain integer counts.
        force_layer (bool, optional): Whether to use specified layer even if not integers in count matrix.
        index (bool, optional): Whether to add extra index or not. Default is True.
        qc_width (float, optional): Width for quality control metric columns. Default is 3.0.
        fontsize (int, optional): Font size for all text in the table. Default is 10.
        savepath (Union[str, os.PathLike, Path], optional): Path to save the figure. Defaults to None.
        save_only (bool, optional): If True, only save without displaying. Defaults to False.
        dpi_save (int, optional): DPI for saved figure. Defaults to 300.

    Raises:
        ImportError: If the 'plottable' framework is not installed.

    Returns:
        None: Displays a plot with the overview table.
    """

    try:
        from plottable import ColumnDefinition, Table
    except ImportError:
        raise ImportError("This function requires the 'plottable' framework. Please install it with 'pip install plottable'.")

    # Copy the metadata, select the columns to plot, and add index if necessary
    df = data._metadata.copy()[columns_to_plot]

    colname_tmp = "ind_tmp"
    if not index and df.shape[1] > 0:
        # Set the first column as the index if index is False
        col_id = df.columns[0]
    else:
        # Rename the index column and reset the index
        df = df.rename_axis(colname_tmp).reset_index()
        col_id = colname_tmp

    # Calculate the maximum cell widths and the total width
    width_dict, total_width = _calculate_max_cell_widths_and_sum(df)
    column_definition = []
    # Add all desired columns from metadata
    for column_name in df.columns:
        border = None
        if column_name == colname_tmp:
            if index:
                border = "right"
            column_definition.append(
                ColumnDefinition(name=column_name,
                                 textprops={"ha": "center"},
                                 width=0.5,
                                #  width=width_dict[column_name],
                                 title="ID", border=border))
        else:
            column_definition.append(
                ColumnDefinition(name=column_name,
                                 group="metadata",
                                 textprops={"ha": "center"},
                                 width=width_dict[column_name]))

    # Calculate predefined QC metrics
    list_gene_count = []
    list_transcript_count = []
    for _, data in data.iterdata():
        if data.cells.is_empty:
            warnings.warn("Cells were not loaded. Loading cells.")
            data.load_cells()

        # get CellData
        celldata = _get_cell_layer(cells=data.cells, cells_layer=cells_layer)

        m_gene_counts, m_transcript_counts = _calculate_single_metrics(
            celldata.table,
            layer=layer,
            force_layer=force_layer)
        list_gene_count.append(m_gene_counts)
        list_transcript_count.append(m_transcript_counts)

    df["mean_transcript_counts"] = list_transcript_count
    df["mean_gene_counts"] = list_gene_count
    max_genes = df["mean_gene_counts"].max()
    max_transcripts = df["mean_transcript_counts"].max()

    # Add all columns with QC metrics
    column_definition_bars = [
        ColumnDefinition(
            "mean_transcript_counts",
            group="qc_metrics",
            plot_fn=_custom_bar,
            plot_kw={"max": max_transcripts, "fontsize": fontsize},
            title="Median Transcripts per Cell",
            textprops={"ha": "center"},
            width=qc_width, border="left"),
        ColumnDefinition(
            "mean_gene_counts",
            group="qc_metrics",
            plot_fn=_custom_bar,
            plot_kw={"max": max_genes, "fontsize": fontsize},
            title="Median Genes per Cell",
            textprops={"ha": "center"},
            width=qc_width
            )
    ]
    # Create the plot
    fig, ax = plt.subplots(figsize=(total_width + qc_width * 2, len(df) * 0.5 + 0.8))
    plt.rcParams["font.family"] = ["DejaVu Sans"]

    table = Table(
        df,
        index_col=col_id,
        column_definitions=(column_definition + column_definition_bars),
        row_dividers=True,
        footer_divider=True,
        ax=ax,
        textprops={"fontsize": fontsize},
        row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
        col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
        column_border_kw={"linewidth": 1, "linestyle": "-"},
        )

    # save and show figure
    save_and_show_figure(savepath=savepath, fig=fig, save_only=save_only, dpi_save=dpi_save, tight=False)

    #plt.show()

# deprecated version
def plot_overview(*args, **kwargs):
    from insitupy._warnings import plot_functions_deprecations_warning
    plot_functions_deprecations_warning(name="overview")