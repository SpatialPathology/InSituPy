import os
import warnings
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt

from insitupy._core._utils import _get_cell_layer
from insitupy.experiment.data import InSituExperiment
from insitupy.io.plots import save_and_show_figure


def plot_overview(
    data: InSituExperiment,
    cells_layer: Optional[str] = None,
    columns_to_plot: List[str] = [],
    layer: str = None,
    force_layer: bool = False,
    index: bool = True,
    qc_width: float = 4.0,
    savepath: Union[str, os.PathLike, Path] = None,
    save_only: bool = False,
    dpi_save: int = 300
    ):
    """
    Plots an overview table with metadata and quality control metrics.

    Args:
        columns_to_plot (List[str]): List of column names to include in the plot.
        layer (str, optional): The layer of the AnnData object to use for calculations. If None, the function will use the main matrix (adata.X) or the 'counts' layer if the main matrix does not contain integer counts.
        force_layer (bool, optional): Whether to use specifies layer even if not integers in count matrix.
        index (bool, optional): Whether to add extra index or not. Default is True.
        custom_width (float, optional): Custom width for metadata columns. Default is 1.0.
        qc_width (float, optional): Width for quality control metric columns. Default is 4.0.

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
    df = data.metadata.copy()[columns_to_plot]

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
                                 width=width_dict[column_name],
                                 title="Sample", border=border))
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

        m_gene_counts, m_transcript_counts = _calculate_metrics(
            celldata.matrix,
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
            plot_kw={"max": max_transcripts},
            title="Median Transcripts per Cell",
            textprops={"ha": "center"},
            width=qc_width, border="left"),
        ColumnDefinition(
            "mean_gene_counts",
            group="qc_metrics",
            plot_fn=_custom_bar,
            plot_kw={"max": max_genes},
            title="Median Genes per Cell",
            textprops={"ha": "center"},
            width=qc_width
            )
    ]
    # Create the plot
    fig, ax = plt.subplots(figsize=(total_width + qc_width * 2, len(df) * 0.7 + 1))
    plt.rcParams["font.family"] = ["DejaVu Sans"]

    table = Table(
        df,
        index_col=col_id,
        column_definitions=(column_definition + column_definition_bars),
        row_dividers=True,
        footer_divider=True,
        ax=ax,
        row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
        col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
        column_border_kw={"linewidth": 1, "linestyle": "-"},
        )

    # save and show figure
    save_and_show_figure(savepath=savepath, fig=fig, save_only=save_only, dpi_save=dpi_save, tight=False)

    #plt.show()

