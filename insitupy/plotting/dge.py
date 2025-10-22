from numbers import Number
from typing import List

import matplotlib.pyplot as plt

from insitupy.dataclasses.results import DiffExprResults


def dual_foldchange_plot(
    results: DiffExprResults,
    significance_threshold: Number,
    fold_change_threshold: Number,
    logfc_col: str = "log2FoldChange",
    pval_col: str = "pvalue",
    patch_colors: List[str] = ["lightgreen", "lightyellow", "lightcoral"],
    adjust_labels: bool = True,
):
    """
    Create volcano-style scatter plots comparing log2 fold changes between
    main DGE results (condition A vs B) and neighbor-based comparisons.

    Parameters
    ----------
    results : PseudobulkDGEResults
        Result container holding main and neighborhood differential expression data.
        Must include both `nb_condition_a` and `nb_condition_b`.
    significance_threshold : Number
        P-value threshold for significance.
    fold_change_threshold : Number
        Minimum absolute log2 fold change to include in the plot.
    logfc_col : str, default="log2FoldChange"
        Column name for log2 fold change values.
    pval_col : str, default="pvalue"
        Column name for p-values.
    patch_colors : list of str, default=["lightgreen", "lightyellow", "lightcoral"]
        Background colors for positive, neutral, and negative regions.
    adjust_labels : bool, default=True
        Whether to automatically adjust overlapping gene labels.

    Returns
    -------
    (fig, axs) : tuple
        Matplotlib figure and axes.
    """
    if not results.has_neighbors():
        raise ValueError("The provided results object has no neighbor comparisons.")

    if adjust_labels:
        try:
            from adjustText import adjust_text
        except ImportError:
            raise ImportError(
                "The 'adjustText' package is required for label adjustment. "
                "Install it via `pip install adjusttext` or set adjust_labels=False."
            )

    # Extract required DataFrames
    df_main = results.main
    df_nb_first = results.nb_condition_a
    df_nb_second = results.nb_condition_b

    # Filter for genes above/below log2FC threshold in main comparison
    filtered_data_x_up = df_main[df_main[logfc_col] >= fold_change_threshold].copy()
    filtered_data_y_up = df_nb_first[df_nb_first.index.isin(filtered_data_x_up.index)]

    filtered_data_x_down = df_main[df_main[logfc_col] <= -fold_change_threshold].copy()
    filtered_data_y_down = df_nb_second[df_nb_second.index.isin(filtered_data_x_down.index)]

    # Create figure
    fig, axs = plt.subplots(1, 2, figsize=(7 * 2, 7))
    xs = [filtered_data_x_down, filtered_data_x_up]
    ys = [filtered_data_y_down, filtered_data_y_up]
    fcs = [-fold_change_threshold, fold_change_threshold]

    for i, (x, y, fc) in enumerate(zip(xs, ys, fcs)):
        x_values = x[logfc_col]
        y_values = y[logfc_col]
        p_values = x[pval_col]
        sig = p_values < significance_threshold
        genes = x.index

        _plot_single_nb_plot(
            x_values=x_values,
            y_values=y_values,
            genes=genes,
            p_values=p_values,
            fold_change_threshold=fc,
            significance_threshold=significance_threshold,
            sig=sig,
            patch_colors=patch_colors,
            ax=axs[i],
            show_legend=(i == 1),
            show_ylabel=(i == 0),
            adjust_labels=adjust_labels
        )

    plt.tight_layout()
    plt.show()
    return fig, axs


def _plot_single_nb_plot(
    x_values,
    y_values,
    genes,
    p_values,
    fold_change_threshold,
    significance_threshold,
    sig,
    patch_colors,
    ax,
    show_legend: bool,
    show_ylabel: bool,
    adjust_labels: bool
):
    """Helper for plotting one volcano-neighbor comparison subplot."""

    if adjust_labels:
        try:
            from adjustText import adjust_text
        except ImportError:
            raise ImportError("The 'adjustText' module is required for label adjustment. Please install it with `pip install adjusttext` or select adjust_labels=False.")

    xmin, xmax = x_values.min(), x_values.max()
    ymin, ymax = y_values.min(), y_values.max()
    ax.set_xlim(xmin - 0.1 * abs(xmin), xmax + 0.1 * abs(xmax))
    ax.set_ylim(ymin - 0.1 * abs(ymin), ymax + 0.1 * abs(ymax))

    # Background patches
    ax.axhspan(0, ymax, facecolor=patch_colors[0], alpha=0.3)
    ax.axhspan(-1, 0, facecolor=patch_colors[1], alpha=0.3)
    if ymin < -1:
        ax.axhspan(ymin, -1, facecolor=patch_colors[2], alpha=0.3)

    # Scatter significant and non-significant points
    ax.scatter(x_values[sig], y_values[sig], c="black", label="significant", alpha=1.0, s=20)
    ax.scatter(x_values[~sig], y_values[~sig], c="gray", label="not significant", alpha=1.0, s=15)

    # Reference lines
    objects_to_avoid = []
    objects_to_avoid.append(ax.axhline(0, color="black", linestyle="--", linewidth=1))
    objects_to_avoid.append(ax.axhline(-1, color="black", linestyle="--", linewidth=1))
    objects_to_avoid.append(ax.axvline(fold_change_threshold, color="black", linestyle="--", linewidth=1))

    # Labels
    texts = []
    for x, y, gene, pval in zip(x_values, y_values, genes, p_values):
        color = "black" if pval < significance_threshold else "gray"
        style = None if pval < significance_threshold else "oblique"
        t = ax.text(x, y, gene, fontsize=10 if style else 12, color=color, fontstyle=style or "normal")
        texts.append(t)

    # Adjust label overlap
    if adjust_labels:
        adjust_text(
            texts,
            ax=ax,
            objects=objects_to_avoid,
            only_move={"text": "xy"},
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
        )

    # Axis labels and legend
    if show_legend:
        ax.legend(title="pvals (target_vs_reference)", loc="center left", bbox_to_anchor=(1, 0.5))
    if show_ylabel:
        ax.set_ylabel("Log2-fold change target_vs_neighbors")
    ax.set_xlabel("Log2-fold change target_vs_reference")
