import logging
from pathlib import Path

import anndata
import pandas as pd

logger = logging.getLogger(__name__)


class XeniumPanels:
    '''
    Class containing a collection of Xenium Panels
    '''
    def __init__(self, verbose=False):
        # read all panels
        # script_dir = Path(os.path.realpath(__file__)).parent
        # panel_dir = script_dir / Path("../xenium_panels/")
        # panel_dir = panel_dir.resolve()
        script_dir = Path(__file__).parent
        panel_dir = (script_dir / "../_xenium_panels/").resolve()
        panel_paths = sorted(panel_dir.glob("[!.]*.csv"))

        for p in panel_paths:
            name = p.stem.split("_")[1]
            panel = pd.read_csv(p)
            setattr(self, name, panel)
            logger.debug(name) if verbose else None

            # make sure that the column names are correct
            panel.columns = ["Gene", "Ensembl_ID", "Coverage", "Codewords", "Annotation"]

    def show_all(self):
        '''
        Prints all available panels.
        '''
        panel_list = [elem for elem in dir(self) if not elem.startswith("__")]
        panel_list = [elem for elem in panel_list if elem not in ["show_all"]]
        for p in panel_list:
            logger.info(p)


def generate_mock_reference(
    dataframe: pd.DataFrame,
    annotation_column: str = "Annotation",
    gene_column: str = "Gene"
    ):
    """
    Generate a mock single-cell reference AnnData from a marker-gene table.

    Creates a synthetic AnnData where each cell type has idealized expression:
    all marker genes for a given cell type are set to 1000 counts and all
    other genes are set to 0. Useful for testing cell-type annotation pipelines.

    Args:
        dataframe (pd.DataFrame): DataFrame with at least two columns: one for
            cell type / annotation labels and one for gene names. Each row
            associates one gene with one annotation.
        annotation_column (str, optional): Name of the column containing
            cell type / annotation labels. Defaults to ``'Annotation'``.
        gene_column (str, optional): Name of the column containing gene names.
            Defaults to ``'Gene'``.

    Returns:
        AnnData: A synthetic reference AnnData with shape
            ``(n_cell_types, n_genes)``. ``obs`` contains an ``'Annotation'``
            column with cell type labels; ``var_names`` are gene names.
    """

    # Initialize an empty dictionary to store marker genes for each annotation
    annotation_genes = {}

    # Iterate over unique annotations
    for annotation in dataframe[annotation_column].unique():
        # Collect marker genes for the annotation
        marker_genes = dataframe.loc[dataframe[annotation_column] == annotation, gene_column].tolist()
        # Add the annotation and its marker genes to the dictionary
        annotation_genes[annotation] = marker_genes

    # Create a DataFrame to store the count matrix
    count_matrix = pd.DataFrame(0, columns=dataframe[gene_column], index=dataframe[annotation_column].unique())

    # Set marker genes to 1000 in the count matrix
    for annotation, genes in annotation_genes.items():
        count_matrix.loc[annotation, genes] = 1000

    # Create annotation DataFrame
    annotation_df = pd.DataFrame({'Annotation': count_matrix.index})

    # match index
    annotation_df.index = count_matrix.index
    # create adata
    reference = anndata.AnnData(X = count_matrix, obs = annotation_df)

    return reference
