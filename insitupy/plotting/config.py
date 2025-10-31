from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties

from insitupy.dataclasses.results import DiffExprConfigCollector

TABLE_ROW_HEIGHT = 0.075

def _add_config_table(
    config: DiffExprConfigCollector,
    n_upreg: int,
    n_downreg: int,
    ax: plt.Axes,
    position: Literal["below", "inside"] = "below"
) -> None:
    """
    Add a configuration summary table to a plot, either below or inside the axis.

    Parameters
    ----------
    config : DiffExprConfigCollector
        Configuration object containing analysis metadata.
    n_upreg : int
        Number of significantly upregulated genes.
    n_downreg : int
        Number of significantly downregulated genes.
    ax : matplotlib.axes.Axes
        Axes object to attach the table to.
    position : {'below', 'inside'}, default='below'
        Where to place the table.
    """
    d = config.to_dict()
    config_table = pd.DataFrame(
        {
            "Reference": list(d["Reference"].values()) + [n_downreg],
            "Target": list(d["Target"].values()) + [n_upreg]
        },
        index=list(d["Target"].keys()) + ["# DEGs"]
    ).dropna()

    if 'name' in config_table.index:
        config_table = config_table.loc[['name'] + [i for i in config_table.index if i != 'name']]

    if "metadata" in config_table.index:
        config_table = config_table.drop("metadata")

    config_table = config_table.reset_index().rename(columns={'index': ''})

    n_rows = len(config_table)
    table_height = TABLE_ROW_HEIGHT * (n_rows + 1)

    if position == "below":
        bbox = [-0.12, -table_height - 0.2, 1.12, table_height]
        ax_pos = ax.get_position()
        new_pos = [ax_pos.x0, ax_pos.y0 - 0.35, ax_pos.width, ax_pos.height * 0.7]
        ax.set_position(new_pos)

    elif position == "inside":
        # Center vertically inside the current axis
        y_center = 0.5 - (table_height / 2)
        bbox = [0.05, y_center, 0.9, table_height]
        ax.axis("off")

    else:
        raise ValueError(f"Invalid position '{position}'. Must be 'below' or 'inside'.")

    # Create the table
    table = ax.table(
        cellText=config_table.values,
        colLabels=config_table.columns,
        cellLoc='center',
        colWidths=[0.3, 0.4, 0.4],
        loc='center',
        bbox=bbox
    )

    # Format cells
    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor('lightgrey')
        if (row == 0) or (col == 0):
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))

    table.scale(xscale=2, yscale=1)

    if position == "inside":
        ax.set_title("Configuration Summary", fontsize=12, pad=10)


# def _add_config_table(
#     config: DiffExprConfigCollector,
#     n_upreg: int,
#     n_downreg: int,
#     ax: plt.Axes,
#     position: Literal["below", "inside"] = "below"
# ) -> None:
#     """
#     Add a configuration summary table to the plot.

#     Parameters
#     ----------
#     config : DiffExprConfigCollector
#         Configuration object containing analysis metadata.
#     n_upreg : int
#         Number of significantly upregulated genes.
#     n_downreg : int
#         Number of significantly downregulated genes.
#     ax : matplotlib.axes.Axes
#         Axes object to attach the table to.
#     position : {'below', 'inside'}, default='below'
#         Where to place the table:
#         - 'below' places it under the existing plot (used for volcano plots).
#         - 'right' creates a side panel (used for dual foldchange plots).

#     Returns
#     -------
#     None
#         Modifies the provided axes in place.
#     """
#     d = config.to_dict()
#     config_table = pd.DataFrame(
#         {
#             "Reference": list(d["Reference"].values()) + [n_downreg],
#             "Target": list(d["Target"].values()) + [n_upreg]
#         },
#         index=list(d["Target"].keys()) + ["# DEGs"]
#     ).dropna()

#     if 'name' in config_table.index:
#         config_table = config_table.loc[['name'] + [i for i in config_table.index if i != 'name']]

#     if "metadata" in config_table.index:
#         config_table = config_table.drop("metadata")

#     config_table = config_table.reset_index().rename(columns={'index': ''})

#     n_rows = len(config_table)
#     table_height = TABLE_ROW_HEIGHT * (n_rows + 1)
#     bbox = [-0.12, -table_height - 0.2, 1.12, table_height]

#     if position == "below":
#         # --- Existing behavior ---


#         table = ax.table(
#             cellText=config_table.values,
#             colLabels=config_table.columns,
#             cellLoc='center',
#             colWidths=[0.3, 0.4, 0.4],
#             loc='bottom',
#             bbox=bbox
#         )

#         # Format table
#         for (row, col), cell in table.get_celld().items():
#             cell.set_facecolor('lightgrey')
#             if (row == 0) or (col == 0):
#                 cell.set_text_props(fontproperties=FontProperties(weight='bold'))

#         table.scale(xscale=2, yscale=1)

#         # Adjust axes position for table
#         pos = ax.get_position()
#         new_pos = [pos.x0, pos.y0 - 0.35, pos.width, pos.height * 0.7]
#         ax.set_position(new_pos)

#     elif position == "inside":
#         # --- New panel behavior ---
#         ax.axis("off")
#         table = ax.table(
#             cellText=config_table.values,
#             colLabels=config_table.columns,
#             cellLoc='center',
#             colWidths=[0.3, 0.4, 0.4],
#             loc='center',
#             bbox=bbox
#             #bbox=[0.05, 0.05, 0.9, 0.9]
#         )

#         for (row, col), cell in table.get_celld().items():
#             cell.set_facecolor('lightgrey')
#             if (row == 0) or (col == 0):
#                 cell.set_text_props(fontproperties=FontProperties(weight='bold'))

#         ax.set_title("Configuration Summary", fontsize=12, pad=10)

#     else:
#         raise ValueError(f"Invalid position '{position}'. Must be 'below' or 'right'.")


# def _add_config_table(
#     config: DiffExprConfigCollector,
#     n_upreg: int,
#     n_downreg: int,
#     ax: plt.Axes
# ) -> None:
#     """
#     Add a configuration summary table below the dual foldchange plot.

#     Parameters
#     ----------
#     config : DiffExprConfigCollector
#         Configuration object containing analysis metadata.
#     n_upreg : int
#         Number of significantly upregulated genes.
#     n_downreg : int
#         Number of significantly downregulated genes.
#     ax : matplotlib.axes.Axes
#         Axes object to attach the table to.

#     Returns
#     -------
#     None
#         Modifies the axes in place.
#     """
#     # Create table data from config
#     d = config.to_dict()
#     config_table = pd.DataFrame(
#         {
#             "Reference": list(d["Reference"].values()) + [n_downreg],
#             "Target": list(d["Target"].values()) + [n_upreg]
#         },
#         index=list(d["Target"].keys()) + ["# DEGs"]
#     ).dropna()

#     # Reorder to put 'name' first if present
#     if 'name' in config_table.index:
#         other_indices = [i for i in config_table.index if i != 'name']
#         config_table = config_table.loc[['name'] + other_indices]

#     # Remove metadata if present (often too verbose)
#     if "metadata" in config_table.index:
#         config_table = config_table.drop("metadata")

#     config_table = config_table.reset_index().rename(columns={'index': ''})

#     # Calculate table dimensions
#     n_rows = len(config_table)
#     table_height = TABLE_ROW_HEIGHT * (n_rows + 1)

#     # Create table
#     table = ax.table(
#         cellText=config_table.values,
#         colLabels=config_table.columns,
#         cellLoc='center',
#         colWidths=[0.3, 0.4, 0.4],
#         loc='bottom',
#         bbox=[-0.12, -table_height - 0.2, 1.12, table_height]
#     )

#     # Format table cells
#     for (row, col), cell in table.get_celld().items():
#         cell.set_facecolor('lightgrey')
#         if (row == 0) or (col == 0):
#             cell.set_text_props(fontproperties=FontProperties(weight='bold'))

#     table.scale(xscale=2, yscale=1)

#     # Adjust axes position to make room for table
#     pos = ax.get_position()
#     new_pos = [pos.x0, pos.y0 - 0.35, pos.width, pos.height * 0.7]
#     ax.set_position(new_pos)