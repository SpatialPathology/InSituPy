import warnings

import scanpy as sc
from anndata import AnnData
from matplotlib.axes._axes import Axes

from insitupy._core._checks import is_integer_counts


def _calculate_max_cell_widths_and_sum(df, multiplier=0.2):
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

def _custom_bar(ax: Axes, val: float, max: float, color: str = None, rect_kw: dict = {}):
        """
        Custom function to create a horizontal bar plot.

        Args:
            ax (Axes): The axes on which to plot.
            val (float): The value to plot.
            max (float): The maximum value for the x-axis.
            color (str, optional): The color of the bar.
            rect_kw (dict, optional): Additional keyword arguments for the rectangle.

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
            ax.text(width + 1, rect.get_y() + rect.get_height() / 2, f'{width:.0f}', ha='left', va='center')
        return bar


def _calculate_metrics(adata: AnnData, layer: str = None, force_layer: bool = False):
        """
        Calculate quality control metrics for an AnnData object.

        Args:
            adata (AnnData): Annotated data matrix.
            layer (str, optional): The layer of the AnnData object to use for calculations. If None, the function will use the main matrix (adata.X) or the 'counts' layer if the main matrix does not contain integer counts.

        Returns:
            tuple: A tuple containing the median number of genes by counts and the median total counts.

        Notes:
            - If no raw counts are provided and the main matrix (adata.X) does not contain integer counts, the function will issue a warning and return (0, 0).
        """
        if layer is None:
            if not is_integer_counts(adata.X) and not force_layer:
                if "counts" not in adata.layers.keys() or ("counts" in adata.layers.keys() and not is_integer_counts(adata.layers["counts"])):
                    warnings.warn("No raw counts provided, metrics are set to 0.")
                    return 0, 0
                else:
                    df_cells, _ = sc.pp.calculate_qc_metrics(adata, percent_top=None, layer="counts")
            else:
                df_cells, _ = sc.pp.calculate_qc_metrics(adata, percent_top=None)
        else:
            if not is_integer_counts(adata.layers[layer]) and not force_layer:
                warnings.warn(f"No raw counts provided in layer '{layer}', metrics are set to 0.")
                return 0, 0
            else:
                df_cells, _ = sc.pp.calculate_qc_metrics(adata, percent_top=None, layer=layer)

        return df_cells["n_genes_by_counts"].median(), df_cells["total_counts"].median()

