import anndata
import numpy as np

from .._core._checks import check_sanity


def split_batches(adata, batch, hvg=None, return_categories=False):
    split = []
    batch_categories = adata.obs[batch].unique()
    if hvg is not None:
        adata = adata[:, hvg]
    for i in batch_categories:
        split.append(adata[adata.obs[batch] == i].copy())
    if return_categories:
        return split, batch_categories
    return split

def run_scanorama(adata,
              batch_key: str,
              hvg: bool = False,
              hvg_key: str = 'highly_variable',
              **kwargs
              ):
    """
    Perform Scanorama batch correction on the input AnnData object.

    Scanorama is a batch correction method for single-cell RNA-seq data.
    For more details, see: https://github.com/brianhie/scanorama/

    Args:
        adata (anndata.AnnData):
            Annotated data matrix, where rows correspond to cells and columns correspond to features.
        batch_key (str):
            Batch key specifying the batch labels for each cell in `adata.obs`.
        hvg (bool, optional):
            If True, use highly variable genes for batch correction. Default is False.
        hvg_key (str, optional):
            Key in `adata.var` indicating highly variable genes if `hvg` is True. Default is 'highly_variable'.
        **kwargs:
            Additional keyword arguments to be passed to scanorama.correct_scanpy function.

    Returns:
        anndata.AnnData:
            Annotated data matrix with batch-corrected values.
            The original data remains unchanged, and the batch-corrected values can be accessed using
            `adata.obsm['X_scanorama']`.
    """
    import scanorama

    check_sanity(adata, batch_key, hvg, hvg_key)

    hvg_genes = list(adata.var.index[adata.var[hvg_key]])

    split, categories = split_batches(adata.copy(), batch_key, hvg=hvg_genes, return_categories=True)
    corrected = scanorama.correct_scanpy(split, return_dimred=True, **kwargs)
    corrected = anndata.AnnData.concatenate(
        *corrected, batch_key=batch_key, batch_categories=categories, index_unique=None
    )
    corrected.obsm['X_emb'] = corrected.obsm['X_scanorama']
    # corrected.uns['emb']=True

    # add scanorama results to original adata - make sure to have correct order of obs
    X_scan = corrected.obsm['X_scanorama']
    orig_obs_names = list(adata.obs_names)
    cor_obs_names = list(corrected.obs_names)
    adata.obsm['X_scanorama'] = np.array([X_scan[orig_obs_names.index(o)] for o in cor_obs_names])
    adata.obsm['X_emb'] = adata.obsm['X_scanorama']

    return adata