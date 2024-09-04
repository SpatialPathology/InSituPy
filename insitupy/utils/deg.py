import numpy as np
import pandas as pd

from insitupy.utils.utils import convert_to_list


def create_deg_dataframe(adata, groups=None, key='rank_genes_groups'):
    """
    Create a DataFrame of differentially expressed genes from rank_genes_groups results.
    Can be used to plot a volcano plot using `plot_volcano`.

    This function extracts gene names, log fold changes, and p-values from the
    specified group in the rank_genes_groups results of an AnnData object.
    It also handles zero p-values to avoid log10(0) errors.

    Args:
        adata (AnnData): The AnnData object containing the results.
        groups (str or None): The name of the group to extract results for.
                                   If None, create DataFrames for all groups.
        key (str): The key in adata.uns where the rank_genes_groups results are stored.
                   Defaults to 'rank_genes_groups'.

    Returns:
        dict:
            - Returns a dictionary of DataFrames for all groups,
              where each key is a group name and the value is the corresponding DataFrame.
    """
    results = adata.uns[key]

    if groups is not None:
        groups = convert_to_list(groups)
    else:
        groups = results['names'].dtype.names

    # Create a dictionary of DataFrames for all groups
    volcano_data_dict = {}

    for group in groups:
        volcano_data = pd.DataFrame({
            'gene': results['names'][group],
            'logfoldchanges': results['logfoldchanges'][group],
            'pvals': results['pvals'][group],
            'scores': results['scores'][group],
        })
        # Replace zero p-values with a small value to avoid log10(0)
        volcano_data['pvals'] = volcano_data['pvals'].replace(0, 1e-300)
        volcano_data['neg_log10_pvals'] = -np.log10(volcano_data['pvals'])
        volcano_data_dict[group] = volcano_data

    return volcano_data_dict
