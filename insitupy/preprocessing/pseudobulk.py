import logging
from numbers import Number
from typing import List, Literal, Optional, Union

import anndata as ad
from sklearn.neighbors import radius_neighbors_graph

from insitupy._checks import try_import
from insitupy.dataclasses._utils import _get_cell_layer
from insitupy.experiment.data import InSituExperiment
from insitupy.utils.utils import convert_to_list

logger = logging.getLogger(__name__)


def get_neighborhood(
    exp: InSituExperiment,
    sample_col: str,
    cells_layer: Optional[str] = None,
    radius: Number = 30
    ):
    """
    Compute spatial neighborhood connectivity matrices for each sample in an experiment.

    For each sample, builds a radius-based neighbor graph from the cell spatial
    coordinates stored in ``adata.obsm['spatial']``.

    Args:
        exp (InSituExperiment): Experiment object containing multiple samples.
        sample_col (str): Name of the ``obs`` column that stores the sample
            identifier, used to key the returned dictionary.
        cells_layer (Optional[str], optional): Name of the cell segmentation
            layer to use. Defaults to None (main layer).
        radius (Number, optional): Radius (in the same units as the spatial
            coordinates) within which cells are considered neighbors.
            Defaults to 30.

    Returns:
        dict: Mapping from sample identifier to a sparse connectivity matrix
            (``scipy.sparse`` CSR) of shape ``(n_cells, n_cells)``.
    """
    matrices = {}

    for id, data in exp.iterdata():
        layer = _get_cell_layer(cells=data.cells, cells_layer=cells_layer)
        coords = layer.table.obsm["spatial"]
        A = radius_neighbors_graph(coords, radius=radius, mode="connectivity", include_self=False)
        matrices[layer.table.obs[sample_col].unique().tolist()[0]] = A

    return matrices


def neighborhoods_pseudobulk(
    adata,
    coords,
    celltype_col,
    counts_layer,
    sample_col,
    radius: int = 20,
    mode: Literal["sum", "mean", "median"] = "sum"
    ):
    """
    Create pseudobulk profiles for the spatial neighborhood of each cell type.

    For each cell type in ``adata.obs[celltype_col]``, identifies all cells that
    are within ``radius`` of at least one cell of that type, then aggregates
    their counts into a pseudobulk profile.

    Requires ``decoupler`` (``pip install decoupler``).

    Args:
        adata: AnnData object containing cell expression and metadata.
        coords: Spatial coordinates array of shape ``(n_cells, 2)``, typically
            from ``adata.obsm['spatial']``.
        celltype_col (str): Name of the ``obs`` column containing cell type labels.
        counts_layer (str): Name of the AnnData layer containing raw counts to
            aggregate. Pass ``None`` to use ``adata.X``.
        sample_col (str): Name of the ``obs`` column containing the sample
            identifier, used by ``decoupler.pp.pseudobulk``.
        radius (int, optional): Neighborhood radius in the same units as
            ``coords``. Defaults to 20.
        mode (Literal['sum', 'mean', 'median'], optional): Aggregation method
            for pseudobulk creation. Defaults to ``'sum'``.

    Returns:
        AnnData: Concatenated pseudobulk profiles with one observation per cell
            type, labeled ``'<sample>_<celltype>_neighbors'`` in
            ``obs_names``, and ``celltype_col`` stored in ``obs``.
    """
    dc = try_import("decoupler", installation_command="pip install decoupler")

    # get AnnData and retrieve coordinates
    # adata = celldata.table
    # coords = celldata.table.obsm["spatial"]

    A = radius_neighbors_graph(
        coords,
        radius=radius,
        mode="connectivity",
        include_self=False)

    celltype_pdatas = {}
    for celltype in adata.obs[celltype_col].unique():
        # select cells of that cell type
        mask = (adata.obs[celltype_col] == celltype).values

        # check which of the neighboring cells is neighbor to at least one cell of that type
        #any_mask = A[mask].any(axis=0)
        #any_mask = A[mask].getnnz(axis=0) > 0 # use sparse methods to save memory
        any_mask = A[mask].astype(bool).sum(axis=0).A1 > 0 # use sparse methods to save memory

        # filter for such neighboring cells
        filtered = adata[any_mask]

        # generate pseudobulk for neighboring cells of the current cell type
        pdata = dc.pp.pseudobulk(
            adata=filtered,
            sample_col=sample_col,
            groups_col=None,
            layer=counts_layer,
            mode=mode
            )

        # make obs_names unique
        if pdata.shape[0] != 1:
            raise ValueError("Pseudobulk AnnData should have only one observation at this point.")
        pdata.obs_names = [f"{str(pdata.obs_names[0])}_{celltype}_neighbors"]
        #pdata.obs_names = [f"{str(pdata.obs_names[0])}_{celltype}"]

        # collect pseudobulks
        celltype_pdatas[celltype] = pdata

    pdata_big = ad.concat(celltype_pdatas, label=celltype_col)
    return pdata_big


def _check_transfer(metadata, columns):
    for col in columns:
        cols_not_found = []
        if col not in metadata.columns:
            cols_not_found.append(col)

        if len(cols_not_found) > 0:
            raise ValueError(f"Column(s) '{', '.join(cols_not_found)}' not found in `.metadata`.")


def pseudobulk(
    exp,
    celltype_col: str,
    cells_layer: Optional[str] = None,
    counts_layer: Optional[str] = None,
    uid_col: str = "uid",
    mode: Literal["sum", "mean", "median"] = "sum",
    calculate_neighbors: bool = False,
    neighbors_radius: int = 20,
    metadata_to_transfer: Union[List[str], str] = None,
    **kwargs
    ):
    """
    Aggregate single-cell counts into pseudobulk profiles per cell type and sample.

    For each sample in ``exp``, groups cells by ``celltype_col`` and aggregates
    their counts using ``decoupler.pp.pseudobulk``. Optionally also computes
    neighborhood-based pseudobulks (see ``calculate_neighbors``).

    Requires ``decoupler`` (``pip install decoupler``).

    Args:
        exp (InSituExperiment): Experiment object containing multiple samples.
        celltype_col (str): Name of the ``obs`` column containing cell type labels.
        cells_layer (Optional[str], optional): Name of the cell segmentation
            layer to use. Defaults to None (main layer).
        counts_layer (Optional[str], optional): Name of the AnnData layer
            containing raw counts to aggregate. Defaults to None (uses ``adata.X``).
        uid_col (str, optional): Column in the experiment metadata that uniquely
            identifies each sample. Defaults to ``'uid'``.
        mode (Literal['sum', 'mean', 'median'], optional): Aggregation function
            for pseudobulk creation. Defaults to ``'sum'``.
        calculate_neighbors (bool, optional): If True, also compute
            neighborhood pseudobulks via
            :func:`~insitupy.preprocessing.pseudobulk.neighborhoods_pseudobulk`.
            Defaults to False.
        neighbors_radius (int, optional): Radius used for spatial neighborhood
            computation when ``calculate_neighbors=True``. Defaults to 20.
        metadata_to_transfer (Union[List[str], str], optional): Column name(s)
            from the experiment metadata to copy into the pseudobulk
            ``obs`` table. Defaults to None.
        **kwargs: Additional keyword arguments forwarded to
            :func:`~insitupy.preprocessing.pseudobulk.neighborhoods_pseudobulk`
            when ``calculate_neighbors=True``.

    Returns:
        AnnData: Pseudobulk AnnData with one observation per
            ``(sample, cell_type)`` combination when ``calculate_neighbors=False``.

        Tuple[AnnData, AnnData]: ``(pdata_final, nb_pdata_final)`` when
            ``calculate_neighbors=True``, where the second element contains
            neighborhood pseudobulks.
    """
    dc = try_import("decoupler", installation_command="pip install decoupler")

    if metadata_to_transfer is not None:
        metadata_to_transfer = convert_to_list(metadata_to_transfer)

        # check if all columns are in the metadata
        _check_transfer(exp._metadata, metadata_to_transfer)

    pdatas = {}
    nb_pdatas = {}
    for meta, data in exp.iterdata():
        # get UID
        uid = meta[uid_col]

        # extract anndata
        celldata= _get_cell_layer(cells=data.cells, cells_layer=cells_layer)
        adata = celldata.table

        # add batch information
        adata.obs["uid"] = uid

        # create pseudobulk from anndata
        pdata = dc.pp.pseudobulk(
            adata=adata,
            sample_col="uid",
            groups_col=celltype_col,
            layer=counts_layer,
            mode=mode,
            )

        # transfer metadata
        pdata = _transfer_metadata(
            pdata,
            meta,
            metadata_to_transfer
            )

        # collect data
        pdatas[uid] = pdata

        if calculate_neighbors:
            pdata_neighbors = neighborhoods_pseudobulk(
                adata=adata,
                coords=adata.obsm["spatial"],
                celltype_col=celltype_col,
                counts_layer=counts_layer,
                sample_col="uid",
                radius=neighbors_radius,
                mode=mode,
                **kwargs
            )

            # transfer metadata
            pdata_neighbors = _transfer_metadata(
                pdata_neighbors,
                meta,
                metadata_to_transfer
                )

            # collect data
            nb_pdatas[uid] = pdata_neighbors

    # concatenate all pseudobulks
    pdata_final = ad.concat(pdatas, label="uid")
    pdata_final.obs["obs_type"] = "cells"

    # Store pseudobulk settings
    settings = dict(
        celltype_col=celltype_col,
        cells_layer=cells_layer,
        counts_layer=counts_layer,
        uid_col=uid_col,
        mode=mode,
        calculate_neighbors=calculate_neighbors,
        neighbors_radius=neighbors_radius,
        metadata_to_transfer=metadata_to_transfer,
        **kwargs
    )
    pdata_final.uns["pseudobulk_settings"] = settings

    if not calculate_neighbors:
        return pdata_final
    else:
        nb_pdata_final = ad.concat(nb_pdatas, label="uid")
        nb_pdata_final.obs["obs_type"] = "neighbors"
        nb_pdata_final.uns["pseudobulk_settings"] = settings
        return pdata_final, nb_pdata_final

def _transfer_metadata(
    pdata,
    meta,
    metadata_to_transfer
    ):
    # transfer metadata
    if metadata_to_transfer is not None:
        metadata_to_transfer = convert_to_list(metadata_to_transfer)

        for col in metadata_to_transfer:
            if col not in pdata.obs.columns:
                pdata.obs[col] = meta[col]
            else:
                logger.warning("Column '%s' already exists in the pseudobulk AnnData.obs. Skipping transfer of this metadata column.", col)
    return pdata

