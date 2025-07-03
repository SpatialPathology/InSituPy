# def plot_binned_expression(
#     self,
#     genes: Union[List[str], str],
#     maxcols: int = 4,
#     figsize: Tuple[int, int] = (8,6),
#     savepath: Union[str, os.PathLike, Path] = None,
#     save_only: bool = False,
#     dpi_save: int = 300,
#     show: bool = True,
#     fontsize: int = 28
#     ):
#     # extract binned expression matrix and gene names
#     binex = self._cells["main"].matrix.varm["binned_expression"]
#     gene_names = self._cells["main"].matrix.var_names

#     genes = convert_to_list(genes)

#     nplots, nrows, ncols = get_nrows_maxcols(len(genes), max_cols=maxcols)

#     # setup figure
#     fig, axs = plt.subplots(nrows, ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows))

#     # scale font sizes
#     plt.rcParams.update({'font.size': fontsize})

#     if nplots > 1:
#         axs = axs.ravel()
#     else:
#         axs = [axs]

#     for i, gene in enumerate(genes):
#         # retrieve binned expression
#         img = binex[gene_names.get_loc(gene)]

#         # determine upper limit for color
#         vmax = np.percentile(img[img>0], 95)

#         # plot expression
#         axs[i].imshow(img, cmap="viridis", vmax=vmax)

#         # set title
#         axs[i].set_title(gene)

#     if nplots > 1:

#         # check if there are empty plots remaining
#         while i < nrows * maxcols - 1:
#             i+=1
#             # remove empty plots
#             axs[i].set_axis_off()

#     if show:
#         fig.tight_layout()
#         save_and_show_figure(savepath=savepath, fig=fig, save_only=save_only, dpi_save=dpi_save)
#     else:
#         return fig, axs


# def hvg(self,
#         hvg_batch_key: Optional[str] = None,
#         hvg_flavor: Literal["seurat", "cell_ranger", "seurat_v3"] = 'seurat',
#         hvg_n_top_genes: Optional[int] = None,
#         verbose: bool = True
#         ) -> None:
#     """
#     Calculate highly variable genes (HVGs) using specified flavor and parameters.

#     Args:
#         hvg_batch_key (str, optional):
#             Batch key for computing HVGs separately for each batch. Default is None, indicating all samples are considered.
#         hvg_flavor (Literal["seurat", "cell_ranger", "seurat_v3"], optional):
#             Flavor of the HVG computation method. Choose between "seurat", "cell_ranger", or "seurat_v3".
#             Default is 'seurat'.
#         hvg_n_top_genes (int, optional):
#             Number of top highly variable genes to identify. Mandatory if `hvg_flavor` is set to "seurat_v3".
#             Default is None.
#         verbose (bool, optional):
#             If True, print progress messages during HVG computation. Default is True.

#     Raises:
#         ValueError: If `hvg_n_top_genes` is not specified for "seurat_v3" flavor or if an invalid `hvg_flavor` is provided.

#     Returns:
#         None: This method modifies the input matrix in place, identifying highly variable genes based on the specified
#             flavor and parameters. It does not return any value.
#     """

#     if hvg_flavor in ["seurat", "cell_ranger"]:
#         hvg_layer = None
#     elif hvg_flavor == "seurat_v3":
#         hvg_layer = "counts" # seurat v3 method expects counts data

#         # n top genes must be specified for this method
#         if hvg_n_top_genes is None:
#             raise ValueError(f"HVG computation: For flavor {hvg_flavor} `hvg_n_top_genes` is mandatory")
#     else:
#         raise ValueError(f'Unknown value for `hvg_flavor`: {hvg_flavor}. Possible values: {["seurat", "cell_ranger", "seurat_v3"]}')

#     if hvg_batch_key is None:
#         print("Calculate highly-variable genes across all samples using {} flavor...".format(hvg_flavor)) if verbose else None
#     else:
#         print("Calculate highly-variable genes per batch key {} using {} flavor...".format(hvg_batch_key, hvg_flavor)) if verbose else None

#     sc.pp.highly_variable_genes(self._cells["main"].matrix, batch_key=hvg_batch_key, flavor=hvg_flavor, layer=hvg_layer, n_top_genes=hvg_n_top_genes)

