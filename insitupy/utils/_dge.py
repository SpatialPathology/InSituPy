from typing import List, Literal, Optional, Tuple, Union

import numpy as np
from anndata import AnnData

from insitupy._core._checks import _check_assignment
from insitupy._core._utils import _get_cell_layer
from insitupy._core.insitudata import InSituData
from insitupy.utils.utils import convert_to_list


# Define the function
def _check_string_in_assignment(entry, string_to_check):
    split_list = entry.split(" & ")
    return string_to_check in split_list

def _check_list_in_assignment(entry, list_to_check):
    split_list = set(entry.split(" & "))
    return not set(list_to_check).isdisjoint(split_list) # check if the sets are not disjoint - meaning at least one element is found in both

def _select_data_for_dge(
    data: InSituData,
    cells_layer: str,
    annotation_tuple: Optional[Union[Tuple[str, str], Tuple[str, List[str]]]] = None,
    cell_type_tuple: Optional[Tuple[str, str]] = None,
    region_tuple: Optional[Tuple[str, str]] = None,
    force_assignment: bool = False,
    verbose: bool = False
    ) -> AnnData:

    # extract anndata object
    celldata = _get_cell_layer(cells=data.cells, cells_layer=cells_layer)
    adata = celldata.matrix.copy()

    ### REGIONS
    if region_tuple is not None:
        # assign region
        _check_assignment(data=data,
                          cells_layer=cells_layer,
                          key=region_tuple[0],
                          force_assignment=force_assignment,
                          modality="regions",
                          verbose=verbose
                          )

        # select only one region
        s = adata.obsm["regions"][region_tuple[0]]
        region_mask = s.apply(_check_string_in_assignment, string_to_check=region_tuple[1])
        if not np.any(region_mask):
            raise ValueError(f"Region '{region_tuple[1]}' not found in key '{region_tuple[0]}'.")

        if verbose:
            print(f"Restrict analysis to region '{region_tuple[1]}' from key '{region_tuple[0]}'.", flush=True)
        adata = adata[region_mask].copy()

    ### ANNOTATIONS
    if annotation_tuple is not None:
        # check if the annotations need to be assigned first
        _check_assignment(data=data,
                          cells_layer=cells_layer,
                          key=annotation_tuple[0],
                          force_assignment=force_assignment,
                          modality="annotations",
                          verbose=verbose
                          )

        # create mask for filtering
        s = adata.obsm["annotations"][annotation_tuple[0]]
        if isinstance(annotation_tuple[1], str):
            annot_mask = s.apply(_check_string_in_assignment, string_to_check=annotation_tuple[1])
        elif isinstance(annotation_tuple[1], list):
            annot_mask = s.apply(_check_list_in_assignment, list_to_check=annotation_tuple[1])
        else:
            raise ValueError(f'Unknown type for second element of `annotation_tuple`: {type(annotation_tuple[1])}. Must be list or str.')

        if not np.any(annot_mask):
            raise ValueError(f"annotation_name '{annotation_tuple[1]}' not found under annotation_key '{annotation_tuple[0]}'.")

        # do filtering
        if verbose:
            print(f"Restrict analysis to annotation '{annotation_tuple[1]}' from key '{annotation_tuple[0]}'.", flush=True)
        adata = adata[annot_mask].copy()

    ### CELL TYPES
    if cell_type_tuple is not None:
        # create mask for filtering
        if isinstance(cell_type_tuple[1], str):
            cell_type_mask = adata.obs[cell_type_tuple[0]] == cell_type_tuple[1]
        elif isinstance(cell_type_tuple[1], list):
            cell_type_mask = adata.obs[cell_type_tuple[0]].isin(cell_type_tuple[1])
        else:
            raise ValueError(f'Unknown type for second element of `cell_type_tuple`: {type(cell_type_tuple[1])}. Must be list or str.')

        if not np.any(cell_type_mask):
            raise ValueError(f"Cell type '{cell_type_tuple[1]}' not found in .obs column '{cell_type_tuple[0]}'.")

        if verbose:
            print(f"Restrict analysis to cell type '{cell_type_tuple[1]}' from .obs column '{cell_type_tuple[0]}'.", flush=True)
        adata = adata[cell_type_mask].copy()

    return adata


def _substitution_func(
    row,
    annotation_key,
    annotation_name,
    check_for_duplicate_assignment_of_reference,
    reference_name=None,
    ignore_duplicate_assignments=False
    ):
    elem = row[annotation_key]
    # check_reference = False
    # if reference_name is not None:
    #     check_reference = True
    try:
        split_name = elem.split(" & ")
        if annotation_name in split_name:
            if check_for_duplicate_assignment_of_reference:
                if reference_name in split_name:
                    if not ignore_duplicate_assignments:
                        raise ValueError(f"Cell '{row.name}' was found to belong to both the annotation and the reference. To ignore this and use only the annotation assignment, use `ignore_duplicate_assignments=True`. Assignment that was found is: {elem}")
                    else:
                        return annotation_name
                else:
                    return annotation_name
            else:
                return annotation_name
        else:
            # TODO: Here also the reference name must be kept - does it really? Only if both annotation and reference are in the same dataset
            return elem
    except AttributeError:
        return elem