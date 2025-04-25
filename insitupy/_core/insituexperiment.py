import os
import warnings
from copy import deepcopy
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union
from uuid import uuid4

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from tqdm import tqdm

import insitupy
from insitupy._constants import LOAD_FUNCS, MODALITIES, MODALITIES_ABBR
from insitupy._core._checks import _all_obs_names_unique, is_integer_counts
from insitupy._core._utils import _get_cell_layer
from insitupy._core.insitudata import InSituData
from insitupy._core.readers import read_xenium
from insitupy._exceptions import ModalityNotFoundError
from insitupy._textformat import textformat as tf
from insitupy.io.files import check_overwrite_and_remove_if_true
from insitupy.io.plots import save_and_show_figure
from insitupy.utils.utils import (convert_to_list, get_nrows_maxcols,
                                  remove_empty_subplots)


class InSituExperiment:
    """
    Class to manage and analyze multiple spatially resolved single-cell transcriptomics experiments.

    .. figure:: ../_static/img/insituexperiment_overview.svg
       :width: 400px
       :align: right
       :class: dark-light

    The class consists of multiple :class:`~insitudata._core.InSituData` datasets and associated metadata as `pandas.DataFrame`.
    This class provides functionality for managing datasets, performing differential gene expression analysis,
    querying metadata, visualizing data, and saving/loading experiments.

    Attributes:
        data (list): A list of datasets in the experiment.
        metadata (pd.DataFrame): A DataFrame containing metadata for the datasets.
        path (str): The save path of the InSituExperiment object.
        collection (anndata.experimental.AnnCollection): *Experimental feature!* A collection of AnnData objects.

    Methods:
        __repr__():
            Provides a string representation of the InSituExperiment object.
        __getitem__(key):
            Retrieves a subset of the experiment based on the provided key.
        __len__():
            Returns the number of datasets in the experiment.
        collection():
            Returns the collection of AnnData objects.
        data():
            Returns the list of datasets.
        metadata():
            Returns a copy of the metadata DataFrame.
        path():
            Returns the save path of the InSituExperiment object.
        _check_obs_uniqueness(cells_layer=None):
            Checks if observation names are unique across all datasets.
        add(data, mode="insitupy", metadata=None):
            Adds a dataset to the experiment and updates metadata.
        append_metadata(new_metadata, by=None, overwrite=False):
            Appends metadata to the existing InSituExperiment object.
        copy():
            Creates a deep copy of the InSituExperiment object.
        dge(target_id, ref_id=None, target_annotation_tuple=None, ...):
            Performs differential gene expression analysis.
        iterdata():
            Iterates over the metadata rows and corresponding data.
        generate_collection(cells_layer=None, label_col="uid"):
            Generates a collection of AnnData objects.
        get_n_cells(cells_layer=None):
            Returns the total number of cells across all datasets.
        load_all(skip=None):
            Loads all data modalities for all datasets.
        load_annotations():
            Loads annotations for all datasets.
        load_cells():
            Loads cell data for all datasets.
        load_images(names="all", nuclei_type="mip", ...):
            Loads images for all datasets.
        load_regions():
            Loads regions for all datasets.
        load_transcripts(transcript_filename="transcripts.parquet"):
            Loads transcript data for all datasets.
        plot_umaps(cells_layer=None, color=None, title_columns=None, ...):
            Creates a plot with UMAPs of all datasets as subplots.
        query(criteria):
            Queries the experiment based on metadata criteria.
        concat(objs, new_col_name=None):
            Concatenates multiple InSituExperiment objects.
        read(path):
            Reads an InSituExperiment object from a specified folder.
        from_config(config_path, mode="insitupy"):
            Creates an InSituExperiment object from a configuration file.
        from_regions(data, region_key, region_names=None):
            Creates an InSituExperiment object from regions of a dataset.
        remove_history():
            Removes the history of all datasets.
        save(verbose=False, overwrite_metadata=True, **kwargs):
            Saves the InSituExperiment object to its current path.
        saveas(path, overwrite=False, verbose=False, **kwargs):
            Saves the InSituExperiment object to a specified path.
        show(index, return_viewer=True):
    """

    from ._deprecated import plot_overview
    def __init__(self):
        """
        Initialize an InSituExperiment object.

        """
        self._metadata = pd.DataFrame(columns=['uid', 'slide_id', 'sample_id'])
        self._data = []
        self._path = None
        self._collection = None

    def __repr__(self):
        """Provide a string representation of the InSituExperiment object.

        Returns:
            str: A string summarizing the InSituExperiment object.
        """
        # extract metadata
        mdf = self.metadata.copy()
        num_samples = len(mdf)

        # check which modalities are loaded and add information as string to the copied metadata dataframe
        loaded_list = []
        for _, data in self.iterdata():
            loaded_modalities = data.get_loaded_modalities()
            loaded_string = "".join(["+" if m in loaded_modalities else "-" for m in MODALITIES])
            loaded_list.append(loaded_string)
        mdf.insert(1, MODALITIES_ABBR, loaded_list)

        # generate string summary
        sample_summary = mdf.to_string(index=True, col_space=4, max_colwidth=15, max_cols=10)
        return (f"{tf.Bold}InSituExperiment{tf.ResetAll} with {num_samples} samples:\n"
                f"{sample_summary}")

    def __getitem__(self, key):
        """Retrieve a subset of the experiment.

        Args:
            key (int, slice, list, np.ndarray, pd.Series): The index, slice, list of indices, boolean mask,
                or Series to retrieve.

        Returns:
            InSituExperiment: A new InSituExperiment object with the selected subset.
        """
        if isinstance(key, int):
            if key > (len(self) - 1):
                raise IndexError(f"Index ({key}) is out of range {len(self)}.")
            key = slice(key, key + 1)

        elif isinstance(key, list):
            if all(isinstance(i, bool) for i in key):
                key = pd.Series(key)
            # If it's a list of indices, we let it pass to iloc below

        elif isinstance(key, pd.Series):
            if key.dtype != bool:
                key = key.tolist()

        # Handle boolean mask
        if isinstance(key, pd.Series) and key.dtype == bool:
            new_experiment = InSituExperiment()
            new_experiment._data = [d for d, k in zip(self._data, key) if k]
            new_experiment._metadata = self._metadata[key].reset_index(drop=True)

        # Handle slices, list of ints, ndarray, or Series of ints
        else:
            new_experiment = InSituExperiment()
            new_experiment._data = [self._data[i] for i in self._metadata.iloc[key].index]
            new_experiment._metadata = self._metadata.iloc[key].reset_index(drop=True)

        # Disconnect object from save path
        new_experiment._path = None
        return new_experiment

    def __len__(self):
        """Return the number of datasets in the experiment.

        Returns:
            int: The number of datasets.
        """
        return len(self._data)

    @property
    def collection(self):
        """Get the collection of anndatas.

        Returns:
            anndata.experimental.AnnCollection: Object that lazily concatenates AnnData objects.
        """
        return self._collection

    @property
    def data(self):
        """Get the dataset dictionary.

        Returns:
            dict: A dictionary of datasets, where keys are sample IDs and values are Dataset objects.
        """
        return self._data

    @property
    def metadata(self):
        """Get the metadata DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing metadata.
        """
        return self._metadata.copy() # the copy prevents the metadata from being modified

    @property
    def path(self):
        """Return save path of the InSituExperiment object.

        Returns:
            str: Save path.
        """
        return self._path

    def _check_obs_uniqueness(
        self,
        cells_layer: Optional[str] = None
        ):
        """
        Check if the observation names are unique across all datasets.
        """
        # get obs dataframes
        obs_list = []
        for _, d in self.iterdata():
            if d.cells is not None:
                celldata = _get_cell_layer(cells=d.cells, cells_layer=cells_layer)
                obs_list.append(celldata.matrix.obs)

        # concatenate the obs dataframes
        all_obs = pd.concat(obs_list, axis=0, ignore_index=False)
        if not all_obs.index.is_unique:
            warnings.warn("Observation names are not unique across all datasets.")

    @property
    def cells(self):
        self.show_modality("cells")

    @property
    def images(self):
        self.show_modality("images")

    @property
    def transcripts(self):
        self.show_modality("transcripts")

    @property
    def annotations(self):
        self.show_modality("annotations")

    @property
    def regions(self):
        self.show_modality("regions")

    def add(self,
            data: Union[str, os.PathLike, Path, InSituData],
            mode: Literal["insitupy", "xenium"] = "insitupy",
            metadata: Optional[dict] = None
            ):
        """Add a dataset to the experiment and update metadata.

        Args:
            dataset (InSituData): A InSituData object to be added.

        Raises:
            TypeError: If the dataset is not an instance of the Dataset class.
        """
        # Check if the dataset is of the correct type
        try:
            data = Path(data)
        except TypeError:
            dataset = data
        else:
            if mode == "insitupy":
                dataset = InSituData.read(data)
            elif mode == "xenium":
                dataset = read_xenium(data)
            else:
                raise ValueError("Invalid mode. Supported modes are 'insitupy' and 'xenium'.")

        from insitupy._core.insitudata import InSituData
        assert dataset.__class__ is InSituData, "Loaded dataset is not an InSituData object."

        # # set a unique ID
        # dataset._set_uid()

        # Add the dataset to the data collection
        self._data.append(dataset)

        # Create a new DataFrame for the new metadata
        new_metadata = {
            'uid': str(uuid4()).split("-")[0],
            'slide_id': dataset.slide_id,
            'sample_id': dataset.sample_id
        }

        if metadata is not None:
            # add information from metadata argument
            new_metadata = metadata | new_metadata

        # convert to dataframe
        new_metadata = pd.DataFrame([new_metadata])

        # Concatenate the new metadata with the existing metadata
        self._metadata = pd.concat([self._metadata, new_metadata], axis=0, ignore_index=True)


    def append_metadata(self,
                        new_metadata: Union[pd.DataFrame, dict, str, os.PathLike, Path],
                        by: Optional[str] = None,
                        overwrite: bool = False
                        ):
        """
        Append metadata to the existing InSituExperiment object.

        Args:
            new_metadata (Union[pd.DataFrame, dict, str, os.PathLike, Path]): The new metadata to be added. Can be a DataFrame, a dictionary, or a path to a CSV/Excel file.
            by (str, optional): The column name to use for pairing metadata. If None, metadata is paired by order.
            overwrite (bool, optional): Whether to overwrite existing columns in the metadata. Defaults to False.

        Raises:
            ValueError: If the 'by' column is not unique in either the existing or new metadata.
        """
        # Convert new_metadata to a DataFrame if it is not already one
        if isinstance(new_metadata, dict):
            new_metadata = pd.DataFrame(new_metadata)
        elif isinstance(new_metadata, (str, os.PathLike, Path)):
            new_metadata = Path(new_metadata)
            if new_metadata.suffix == '.csv':
                new_metadata = pd.read_csv(new_metadata)
            elif new_metadata.suffix in ['.xlsx', '.xls']:
                new_metadata = pd.read_excel(new_metadata)
            else:
                raise ValueError("Unsupported file format. Please provide a path to a CSV or Excel file.")

        # Create a copy of the existing metadata
        old_metadata = self._metadata.copy()

        if overwrite:
            # preserve only the columns of the old metadata that are not in the new metadata
            cols_to_use = list(old_metadata.columns.difference(new_metadata.columns))

            if by is not None:
                cols_to_use = [by] + cols_to_use

                # sort them by the original order
                cols_to_use = [elem for elem in old_metadata.columns if elem in cols_to_use]

            old_metadata = old_metadata[cols_to_use]
        else:
            # preserve only such columns of the new metadata that are not yet in the old metadata
            cols_to_use = list(new_metadata.columns.difference(old_metadata.columns))

            if by is not None:
                cols_to_use = [by] + cols_to_use

            new_metadata = new_metadata[cols_to_use]

        if by is None:
            if len(new_metadata) != len(old_metadata):
                raise ValueError("Length of new metadata does not match the existing metadata.")
            warnings.warn("No 'by' column provided. Metadata will be paired by order.")
            #updated_metadata = pd.concat([updated_metadata.reset_index(drop=True), new_metadata.reset_index(drop=True)], axis=1)
            updated_metadata = pd.merge(left=old_metadata, right=new_metadata,
                                        left_index=True, right_index=True, how="left")
        else:
            if by not in old_metadata.columns or by not in new_metadata.columns:
                raise ValueError(f"Column '{by}' must be present in both existing and new metadata.")

            if not old_metadata[by].is_unique or not new_metadata[by].is_unique:
                raise ValueError(f"Column '{by}' must be unique in both existing and new metadata.")

            updated_metadata = pd.merge(left=old_metadata, right=new_metadata,
                                        on=by, how="left")

        # Ensure the metadata is paired with the correct data
        if len(updated_metadata) != len(self._data):
            raise ValueError("The number of metadata entries does not match the number of data entries.")

        # Update the object's metadata only if the check passes
        self._metadata = updated_metadata

    def copy(self):
        """Create a deep copy of the InSituExperiment object.

        Returns:
            InSituExperiment: A new InSituExperiment object that is a deep copy of the current object.
        """
        for xd in self._data:
            if xd.viewer is not None:
                xd.viewer = None
        return deepcopy(self)

    def dge(self,
            target_id: int,
            ref_id: Optional[Union[int, List[int], Literal["rest"]]] = None,
            target_annotation_tuple: Optional[Tuple[str, str]] = None,
            target_cell_type_tuple: Optional[Tuple[str, str]] = None,
            target_region_tuple: Optional[Tuple[str, str]] = None,
            ref_annotation_tuple: Optional[Union[Literal["rest", "same"], Tuple[str, str]]] = "same",
            ref_cell_type_tuple: Optional[Union[Literal["rest", "same"], Tuple[str, str]]] = "same",
            ref_region_tuple: Optional[Union[Literal["rest", "same"], Tuple[str, str]]] = "same",
            plot_volcano: bool = True,
            method: Optional[Literal['logreg', 't-test', 'wilcoxon', 't-test_overestim_var']] = 't-test',
            exclude_ambiguous_assignments: bool = False,
            force_assignment: bool = False,
            name_col: str = "sample_id",
            title: Optional[str] = None,
            savepath: Union[str, os.PathLike, Path] = None,
            save_only: bool = False,
            dpi_save: int = 300,
            **kwargs
            ):
        """
        Wrapper function for performing differential gene expression analysis within an `InSituExperiment` object.

        This function serves as a wrapper around the `differential_gene_expression` function,
        facilitating the retrieval of data and metadata, and the generation of a plot title
        if not provided. It compares gene expression between specified annotations within
        a single InSituData object or between two InSituData objects.

        Args:
            target_id (int): Index for the target dataset in the `InSituExperiment` object.
            ref_id (Optional[Union[int, List[int], Literal["rest"]]]): Index or list of indices for the reference dataset in the `InSituExperiment` object.
            target_annotation_tuple (Optional[Tuple[str, str]]): Tuple containing the annotation key and name for the primary data.
            target_cell_type_tuple (Optional[Tuple[str, str]]): Tuple specifying an observation key and value to filter the primary data.
            target_region_tuple (Optional[Tuple[str, str]]): Tuple specifying a region key and name to restrict the analysis to a specific region in the primary data.
            ref_annotation_tuple (Optional[Union[Literal["rest", "same"], Tuple[str, str]]]): Tuple containing the reference annotation key and name, or "rest" to use the rest of the data as reference. Defaults to "same".
            ref_cell_type_tuple (Optional[Union[Literal["rest", "same"], Tuple[str, str]]]): Tuple specifying an observation key and value to filter the reference data. Defaults to "same".
            ref_region_tuple (Optional[Union[Literal["rest", "same"], Tuple[str, str]]]): Tuple specifying a region key and name to restrict the analysis to a specific region in the reference data. Defaults to "same".
            plot_volcano (bool, optional): Whether to generate a volcano plot of the results. Defaults to True.
            method (Optional[Literal['logreg', 't-test', 'wilcoxon', 't-test_overestim_var']], optional): Statistical method to use for differential expression analysis. Defaults to 't-test'.
            exclude_ambiguous_assignments (bool, optional): Whether to exclude ambiguous assignments in the data. Defaults to False.
            force_assignment (bool, optional): Whether to force assignment of annotations and regions. Defaults to False.
            name_col (str, optional): Column name in metadata to use for naming samples. Defaults to "sample_id".
            title (Optional[str], optional): Title for the volcano plot. If not provided, a title is generated based on the data and reference names. Defaults to None.
            savepath (Union[str, os.PathLike, Path], optional): Path to save the plot. Defaults to None.
            save_only (bool): If True, only save the plot without displaying it. Defaults to False.
            dpi_save (int): Dots per inch (DPI) for saving the plot. Defaults to 300.
            **kwargs: Additional keyword arguments to pass to the `differential_gene_expression` function.

        Returns:
            None

        Example:
            >>> analysis.dge(
                    target_id=1,
                    ref_id=2,
                    target_annotation_tuple=("cell_type", "neuron"),
                    ref_annotation_tuple=("cell_type", "astrocyte"),
                    plot_volcano=True,
                    method='wilcoxon'
                )
        """
        from insitupy.tools.dge import differential_gene_expression

        # get data and extract information about experiment
        target = self.data[target_id]
        target_name = self.metadata.loc[target_id, name_col]

        if ref_id is not None:
            if ref_id == "rest":
                ref = [d for i, (m, d) in enumerate(self.iterdata()) if i != target_id]
                ref_name = [m[name_col] for i, (m, d) in enumerate(self.iterdata()) if i != target_id]
                ref_name = ", ".join(ref_name)

            elif isinstance(ref_id, int):
                ref = self.data[ref_id]
                ref_name = self.metadata.loc[ref_id, name_col]
            elif isinstance(ref_id, list):
                ref = [self.data[i] for i in ref_id]
                ref_name = [self.metadata.iloc[i][name_col] for i in ref_id]
                ref_name = ", ".join(ref_name)
            else:
                raise ValueError(f"Argument `ref_id` has to be either int, list of int or 'rest'. Instead: {ref_id}")

        else:
            ref = None
            ref_name = target_name

        title = f"{target_name} vs. {ref_name}"

        dge_res = differential_gene_expression(
            target=target,
            ref=ref,
            target_annotation_tuple=target_annotation_tuple,
            target_cell_type_tuple=target_cell_type_tuple,
            target_region_tuple=target_region_tuple,
            ref_annotation_tuple=ref_annotation_tuple,
            ref_cell_type_tuple=ref_cell_type_tuple,
            ref_region_tuple=ref_region_tuple,
            plot_volcano=plot_volcano,
            method=method,
            exclude_ambiguous_assignments=exclude_ambiguous_assignments,
            force_assignment=force_assignment,
            title = title,
            savepath = savepath,
            save_only = save_only,
            dpi_save = dpi_save,
            **kwargs
        )
        if not plot_volcano:
            return dge_res

    def iterdata(self):
        """Iterate over the metadata rows and corresponding data.

        Yields:
            tuple: A tuple containing the index, metadata row as a Series, and the corresponding data.
        """
        for idx, row in self._metadata.iterrows():
            yield row, self._data[idx]

    def generate_collection(
        self,
        cells_layer: Optional[str],
        label_col: str = "uid"
        ):
        from anndata.experimental import AnnCollection

        adatas = {}
        for meta, xd in self.iterdata():
            celldata = _get_cell_layer(cells=xd.cells, cells_layer=cells_layer)
            adatas[meta[label_col]] = celldata.matrix

        self._collection = AnnCollection(adatas,
                                         join_vars='inner', join_obs='inner',
                                         label=label_col)

    def get_n_cells(
        self,
        cells_layer: Optional[str] = None
        ):
        n_cells = 0
        for _, d in self.iterdata():
            if d.cells is not None:
                celldata = _get_cell_layer(cells=d.cells, cells_layer=cells_layer)
                n_cells += len(celldata.matrix)

        return n_cells

    def load_all(self,
                 skip: Optional[str] = None,
                 ):
        for xd in tqdm(self._data):
            for f in LOAD_FUNCS:
                if skip is None or skip not in f:
                    func = getattr(xd, f)
                    try:
                        func()
                    except ModalityNotFoundError as err:
                        print(err)

    def load_annotations(self):
        for xd in tqdm(self._data):
            xd.load_annotations()

    def load_cells(self):
        for xd in tqdm(self._data):
            xd.load_cells()

    def load_images(self,
                    names: Union[Literal["all", "nuclei"], str] = "all", # here a specific image can be chosen
                    nuclei_type: Literal["focus", "mip", ""] = "mip",
                    load_cell_segmentation_images: bool = True
                    ):

        for xd in tqdm(self._data):
            xd.load_images(names=names,
                           nuclei_type=nuclei_type,
                           load_cell_segmentation_images=load_cell_segmentation_images)

    def load_regions(self):
        for xd in tqdm(self._data):
            xd.load_regions()

    def load_transcripts(self,
                        transcript_filename: str = "transcripts.parquet"
                        ):
        for xd in tqdm(self._data):
            xd.load_transcripts()

    # def make_obs_names_unique(self,
    #                           cells_layer: Optional[str],
    #                           force: bool = False):

    #     if not _all_obs_names_unique(exp=self, cells_layer=cells_layer) or force:
    #         print(f"Make `obs_names` unique.")
    #         for meta, data in self.iterdata():
    #             celldata = _get_cell_layer(cells=data.cells, cells_layer=cells_layer)

    #             # generate new, unique names
    #             new_names = f'{meta["uid"]}-' + celldata.matrix.obs_names
    #             new_names = new_names.astype(str)
    #             return new_names

    #             if not len(new_names) == len(np.unique(new_names)):
    #                 raise ValueError("New names are not unique.")

    #             # add new names to matrix and boundaries
    #             celldata.matrix.obs_names = new_names
    #             celldata.boundaries._cell_names = da.from_array(new_names.astype(str))
    #     else:
    #         print(f"The `obs_names` in samples within the InSituExperiment are already unique. Skipped execution. To force the execution set `force=True`.")

    def plot_umaps(self,
                   cells_layer: Optional[str] = None,
                   color: Optional[str] = None,
                   title_columns: Optional[Union[List[str], str]] = None,
                   title_size: int = 20,
                   max_cols: int = 4,
                   figsize: Tuple[int, int] = (8,6),
                   savepath: Optional[Union[str, os.PathLike, Path]] = None,
                   save_only: bool = False,
                   show: bool = True,
                   fig: Optional[Figure] = None,
                   dpi_save: int = 300,
                   **kwargs):
        """Create a plot with UMAPs of all datasets as subplots using scanpy's pl.umap function.

        Args:
            color (str, optional): Keys for annotations of observations/cells or variables/genes to color the plot. Defaults to None.
            title_columns (str or list of str, optional): List of column names from metadata to use for subplot titles. Defaults to None.
            max_cols (int, optional): Maximum number of columns for subplots. Defaults to 4.
            **kwargs: Additional keyword arguments to pass to sc.pl.umap.
            figsize (tuple, optional): Figure size. Defaults to (8, 6).
            savepath (optional): Path to save the plot.
            save_only (bool, optional): Whether to only save the plot without showing. Defaults to False.
            show (bool, optional): Whether to show the plot. Defaults to True.
            fig (optional): Figure to plot on.
            dpi_save (int, optional): DPI for saving the plot. Defaults to 300.
        """
        num_datasets = len(self._data)
        n_plots, n_rows, max_cols = get_nrows_maxcols(len(self._data), max_cols)
        fig, axes = plt.subplots(n_rows, max_cols, figsize=(figsize[0]*max_cols, figsize[1]*n_rows))
        if n_plots > 1:
            axes = axes.ravel()

        # make sure title_columns is a list
        if title_columns is not None:
            title_columns = convert_to_list(title_columns)

        for idx, (metadata_row, dataset) in enumerate(self.iterdata()):
            ax = axes[idx] if num_datasets > 1 else axes

            # Get data from MultiCellData
            celldata = _get_cell_layer(cells=dataset.cells, cells_layer=cells_layer)
            adata = celldata.matrix

            # plot UMAP and add to axis
            sc.pl.umap(adata, ax=ax, color=color, show=False, **kwargs)

            if title_columns:
                title = " - ".join(str(metadata_row[col]) for col in title_columns if col in metadata_row)
                ax.set_title(title, fontdict={"fontsize": title_size})
            else:
                ax.set_title(f"Dataset {idx + 1}", fontdict={"fontsize": title_size})

        remove_empty_subplots(
            axes, n_plots, n_rows, max_cols
        )
        if show:
            save_and_show_figure(savepath=savepath, fig=fig, save_only=save_only, dpi_save=dpi_save, tight=True)
        else:
            return fig, axes

    def query(self, criteria):
        """Query the experiment based on metadata criteria.

        Args:
            criteria (dict or str):
                - A dictionary where keys are column names and values are lists of categories to select.
                - A string expression to evaluate using pandas.DataFrame.query().

        Returns:
            InSituExperiment: A new InSituExperiment object with the selected subset.
        """
        if isinstance(criteria, dict):
            mask = pd.Series([True] * len(self._metadata), index=self._metadata.index)
            for column, values in criteria.items():
                values = convert_to_list(values)
                if column in self._metadata.columns:
                    mask &= self._metadata[column].isin(values)
                else:
                    raise KeyError(f"Column '{column}' not found in metadata.")
            return self[mask]

        elif isinstance(criteria, str):
            try:
                result_df = self._metadata.query(criteria)
                return self[result_df.index]
            except Exception as e:
                raise ValueError(f"Failed to evaluate query expression: {e}")

        else:
            raise TypeError("Criteria must be either a dictionary or a string.")


    @classmethod
    def concat(cls, objs, new_col_name=None):
        """
        Concatenate multiple InSituExperiment objects.

        Args:
            objs (Union[List[InSituExperiment], Dict[str, InSituExperiment]]):
                A list of InSituExperiment objects or a dictionary where keys are added as a new column.
            new_col_name (str, optional):
                The name of the new column to add when objs is a dictionary. Defaults to None.

        Returns:
            InSituExperiment: A new InSituExperiment object containing the concatenated data.
        """
        if isinstance(objs, dict):
            if new_col_name is None:
                raise ValueError("new_col_name must be provided when objs is a dictionary.")
            keys, objs = zip(*objs.items())
        else:
            keys = [None] * len(objs)

        # Initialize a new InSituExperiment object
        new_experiment = cls()

        # Concatenate data and metadata
        new_data = []
        new_metadata = []

        for key, obj in zip(keys, objs):
            if not isinstance(obj, InSituExperiment):
                raise TypeError("All objects must be instances of InSituExperiment.")
            new_data.extend(obj._data)
            metadata = obj._metadata.copy()
            if key is not None:
                metadata[new_col_name] = key
            new_metadata.append(metadata)

        new_experiment._data = new_data
        new_experiment._metadata = pd.concat(new_metadata, ignore_index=True)

        # Disconnect object from save path
        new_experiment._path = None

        # check if observation names are unique
        new_experiment._check_obs_uniqueness()

        return new_experiment

    @classmethod
    def read(cls, path: Union[str, os.PathLike, Path]):
        """Read an InSituExperiment object from a specified folder.

        Args:
            path (Union[str, os.PathLike, Path]): The path to the folder where datasets are saved.

        Returns:
            InSituExperiment: A new InSituExperiment object with the loaded data.
        """
        path = Path(path)

        # Load metadata
        metadata_path = path / "metadata.csv"
        metadata = pd.read_csv(metadata_path, index_col=0)

        # Load each dataset
        data = []
        dataset_paths = sorted([elem for elem in path.glob("data-*") if elem.is_dir()])
        for dataset_path in tqdm(dataset_paths):
            dataset = InSituData.read(dataset_path)
            data.append(dataset)

        # Create a new InSituExperiment object
        experiment = cls()
        experiment._metadata = metadata
        experiment._data = data
        experiment._path = path

        return experiment

    @classmethod
    def from_config(cls,
                    config_path: Union[str, os.PathLike, Path],
                    mode: Literal["insitupy", "xenium"] = "insitupy",
                    **kwargs
                    ):
        """
        Create an InSituExperiment object from a configuration file.

        Args:
            config_path (Union[str, os.PathLike, Path]): The path to the configuration CSV or Excel file.
            mode (Literal["insitupy", "xenium"], optional): The mode to use for loading the datasets. Defaults to "insitupy".

        The configuration file should be either a CSV or Excel file (.csv, .xlsx, .xls) and must contain the following columns:

        - **directory**: This column is mandatory and should contain the paths to the directories where the datasets are stored. Each path should be a valid directory path.
        - **Other columns**: These columns can contain any additional metadata you want to associate with each dataset. The metadata will be extracted from these columns and stored in the InSituExperiment object.

        Example of a valid configuration file:

        | directory         | experiment_name | date       | patient    |
        |-------------------|-----------------|------------|------------|
        | /path/to/dataset1 | Experiment 1    | 2023-09-01 | Patient A  |
        | /path/to/dataset2 | Experiment 2    | 2023-09-02 | Patient B  |

        Returns:
            InSituExperiment: A new InSituExperiment object with the loaded data and metadata.
        """
        config_path = Path(config_path)

        # Determine file type and read the configuration file
        if config_path.suffix in ['.csv']:
            config = pd.read_csv(config_path, dtype=str)
        elif config_path.suffix in ['.xlsx', '.xls']:
            config = pd.read_excel(config_path, dtype=str)
        else:
            raise ValueError("Unsupported file type. Please provide a CSV or Excel file.")

        # Ensure the 'directory' column exists
        if 'directory' not in config.columns:
            raise ValueError("The configuration file must contain a 'directory' column.")

        # Get the current working directory
        current_path = Path.cwd()

        # Initialize a new InSituExperiment object
        experiment = cls()

        # Iterate over each row in the configuration file
        # for _, row in tqdm(config.iterrows()):
        for i in tqdm(range(len(config))):
            row = config.iloc[i, :]
            dataset_path = Path(row['directory'])

            # Check if the path is relative and if so, append the current path
            if not dataset_path.is_absolute():
                dataset_path = current_path / dataset_path

            # Check if the directory exists
            if not dataset_path.exists():
                raise FileNotFoundError(f"No such directory found: {str(dataset_path)}")

            if mode == "insitupy":
                dataset = InSituData.read(dataset_path)
            elif mode == "xenium":
                dataset = read_xenium(dataset_path, verbose=False, **kwargs)
            else:
                raise ValueError("Invalid mode. Supported modes are 'insitupy' and 'xenium'.")

            experiment._data.append(dataset)

            # Extract metadata from the row, excluding the 'directory' column
            metadata = row.drop(labels=['directory']).to_dict()
            metadata['uid'] = str(uuid4()).split("-")[0]
            metadata['slide_id'] = dataset.slide_id
            metadata['sample_id'] = dataset.sample_id

            # Append the metadata to the experiment's metadata DataFrame
            experiment._metadata = pd.concat([experiment._metadata, pd.DataFrame([metadata])], ignore_index=True)

        return experiment

    @classmethod
    def from_regions(cls,
                    data: InSituData,
                    region_key: str,
                    region_names: Optional[Union[List[str], str]] = None
                    ):

        # Retrieve the regions dataframe
        region_df = data.regions[region_key]

        # check which region names are allowed
        if region_names is None:
            region_names = region_df["name"].tolist()
        else:
            # make sure region_names is a list
            region_names = convert_to_list(region_names)

        # Initialize a new InSituExperiment object
        experiment = cls()

        for n in sorted(region_df["name"].tolist()):
            if n in region_names:
                # crop data by region
                cropped_data = data.crop(region_tuple=(region_key, n))

                # create metadata
                metadata = {"region_key": region_key, "region_name": n}

                # add to InSituExperiment
                experiment.add(data=cropped_data, metadata=metadata)

        return experiment

    def remove_history(self):
        for xd in tqdm(self._data):
            xd.remove_history(verbose=False)

    def save(self,
             verbose: bool = False,
             overwrite_metadata: bool = True,
             **kwargs
             ):
        if self.path is None:
            print("No save path found in `.path`. First save the InSituExperiment using '.saveas()'.")
            return
        else:
            parent_path_identical = [d.path.parent == self.path for d in self.data]
            if not np.all(parent_path_identical):
                print(f"Saving process failed. Save path of some InSituData objects did not lie inside the InSituExperiment save path: {self.metadata['uid'][parent_path_identical].values}")
            else:
                for xd in tqdm(self._data):
                    xd.save(
                        verbose=verbose,
                        **kwargs
                        )

        if overwrite_metadata:
            # Optionally, save the metadata as a CSV file
            metadata_path = os.path.join(self.path, "metadata.csv")
            self._metadata.to_csv(metadata_path, index=True)


    def saveas(self, path: Union[str, os.PathLike, Path],
               overwrite: bool = False,
               verbose: bool = False, **kwargs):
        """Save all datasets to a specified folder.

        Args:
            path (Union[str, os.PathLike, Path]): The path to the folder where datasets will be saved.
        """
        # Create the main directory if it doesn't exist
        path = Path(path)

        # check overwrite
        check_overwrite_and_remove_if_true(path=path, overwrite=overwrite)

        print(f"Saving InSituExperiment to {str(path)}") if verbose else None

        # Iterate over the datasets and save each one in a numbered subfolder
        for index, dataset in enumerate(tqdm(self._data)):
            subfolder_path = path / f"data-{str(index).zfill(3)}"
            dataset.saveas(subfolder_path, verbose=False, **kwargs)

        # Optionally, save the metadata as a CSV file
        metadata_path = os.path.join(path, "metadata.csv")
        self._metadata.to_csv(metadata_path, index=True)

        print("Saved.") if verbose else None

    def show(self, index: int, return_viewer: bool = True):
        """
        Displays the dataset at the specified index.

        Args:
            index (int): The index of the dataset to display.
            return_viewer (bool, optional): If True, returns the viewer object of the dataset. Defaults to True.

        Returns:
            Viewer: The viewer object of the dataset if return_viewer is True.
        """
        dataset = self.data[index]
        dataset.show()
        if return_viewer:
            return dataset.viewer

    def show_modality(self, modality, uid_column: str = "sample_id"):
        repr_string = ""
        for meta, data in self.iterdata():
            repr_string += f"{meta.name}: {tf.Bold+tf.Red}{meta[uid_column]}{tf.ResetAll}\n"
            repr_string += f"{tf.SPACER}   " + data.get_modality(modality).__repr__().replace("\n", f"\n{tf.SPACER}   ") + "\n"

        print(repr_string)