import os
import warnings
from copy import deepcopy
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure

import insitupy
from insitupy import differential_gene_expression
from insitupy._constants import LOAD_FUNCS
from insitupy._core._checks import is_integer_counts
from insitupy._exceptions import ModalityNotFoundError
from insitupy.io.files import check_overwrite_and_remove_if_true
from insitupy.io.plots import save_and_show_figure
from insitupy.utils.utils import (convert_to_list, get_nrows_maxcols,
                                  remove_empty_subplots)
from insitupy.utils.utils import textformat as tf


class InSituExperiment:
    def __init__(self):
        """
        Initialize an InSituExperiment object.

        """
        self._metadata = pd.DataFrame(columns=['uid', 'slide_id', 'sample_id'])
        self._data = []
        self._path = None

    def __repr__(self):
        """Provide a string representation of the InSituExperiment object.

        Returns:
            str: A string summarizing the InSituExperiment object.
        """
        num_samples = len(self._metadata)
        sample_summary = self._metadata.to_string(index=True, col_space=4)
        return (f"{tf.Bold}InSituExperiment{tf.ResetAll} with {num_samples} samples:\n"
                f"{sample_summary}")

    def __getitem__(self, key):
        """Retrieve a subset of the experiment.

        Args:
            key (int, slice, list, pd.Series): The index, slice, list, or boolean Series to retrieve.

        Returns:
            InSituExperiment: A new InSituExperiment object with the selected subset.
        """
        if isinstance(key, int):
            if key > (len(self)-1):
                raise IndexError(f"Index ({key}) is out of range {len(self)}.")
            key = slice(key, key + 1)
        elif isinstance(key, list) and all(isinstance(i, bool) for i in key):
            key = pd.Series(key)
        if isinstance(key, pd.Series) and key.dtype == bool:
            new_experiment = InSituExperiment()
            new_experiment._data = [d for d, k in zip(self._data, key) if k]
            new_experiment._metadata = self._metadata[key].reset_index(drop=True)
        else:
            new_experiment = InSituExperiment()
            new_experiment._data = self._data[key]
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

    def add(self,
            data: Union[str, os.PathLike, Path, insitupy.InSituData],
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
            dataset = insitupy.read_xenium(data)
        assert isinstance(dataset, insitupy.InSituData), "Loaded dataset is not an InSituData object."

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
            data_id: int,
            data_annotation_tuple: Union[Tuple[str, str], Tuple[str, str]], # tuple of annotation key and names
            ref_id: Optional[int] = None,
            ref_annotation_tuple: Union[Literal["rest"], Tuple[str, str], Tuple[str, str]] = "rest",
            obs_tuple: Optional[Tuple[str, str]] = None,
            region_tuple: Optional[Union[Tuple[str, str], Tuple[str, str]]] = None,
            plot_volcano: bool = True,
            method: Optional[Literal['logreg', 't-test', 'wilcoxon', 't-test_overestim_var']] = 't-test',
            ignore_duplicate_assignments: bool = False,
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
            data_id (int): Identifier for the primary dataset within the `InSituExperiment` object.
            data_annotation_tuple (Union[Tuple[str, str], Tuple[str, str]]): Tuple containing the annotation key and name for the primary data.
            ref_id (Optional[int]): Identifier for the reference dataset within the `InSituExperiment` object.
            ref_annotation_tuple (Union[Literal["rest"], Tuple[str, str], Tuple[str, str]], optional): Tuple containing the reference annotation key and name, or "rest" to use the rest of the data as reference. Defaults to "rest".
            obs_tuple (Optional[Tuple[str, str]], optional): Tuple specifying an observation key and value to filter the data. Defaults to None.
            region_tuple (Optional[Union[Tuple[str, str], Tuple[str, str]]], optional): Tuple specifying a region key and name to restrict the analysis to a specific region. Defaults to None.
            plot_volcano (bool, optional): Whether to generate a volcano plot of the results. Defaults to True.
            method (Optional[Literal['logreg', 't-test', 'wilcoxon', 't-test_overestim_var']], optional): Statistical method to use for differential expression analysis. Defaults to 't-test'.
            ignore_duplicate_assignments (bool, optional): Whether to ignore duplicate assignments in the data. Defaults to False.
            force_assignment (bool, optional): Whether to force assignment of annotations and regions. Defaults to False.
            name_col (str, optional): Column name in metadata to use for naming samples. Defaults to "sample_id".
            title (Optional[str], optional): Title for the volcano plot. If not provided, a title is generated based on the data and reference names. Defaults to None.
            savepath (Union[str, os.PathLike, Path], optional): Path to save the plot (default is None).
            save_only (bool): If True, only save the plot without displaying it (default is False).
            dpi_save (int): Dots per inch (DPI) for saving the plot (default is 300).

        Returns:
            None

        Example:
            >>> analysis.dge(
                    data_id=1,
                    ref_id=2,
                    data_annotation_tuple=("cell_type", "neuron"),
                    ref_annotation_tuple=("cell_type", "astrocyte"),
                    plot_volcano=True,
                    method='wilcoxon'
                )
        """

        # get data and extract information about experiment
        data = self.data[data_id]
        data_name = self.metadata.loc[data_id, name_col]

        if ref_id is not None:
            ref_data = self.data[ref_id]
            ref_name = self.metadata.loc[ref_id, name_col]
        else:
            ref_data = None
            ref_name = data_name

        if isinstance(data_annotation_tuple, tuple):
            data_annot_name = data_annotation_tuple[1]
        elif data_annotation_tuple == "rest":
            data_annot_name = data_annotation_tuple
        else:
            raise ValueError(f"Argument `data_annotation_tuple` has to be either tuple or 'rest'. Instead: {data_annotation_tuple}")

        if isinstance(ref_annotation_tuple, tuple):
            ref_annot_name = ref_annotation_tuple[1]
        elif ref_annotation_tuple == "rest":
            ref_annot_name = ref_annotation_tuple
        else:
            raise ValueError(f"Argument `ref_annotation_tuple` has to be either tuple or 'rest'. Instead: {ref_annotation_tuple}")

        # create title if necessary
        if title is None:
            if obs_tuple is not None:
                cell_title_part = f"\n{obs_tuple[0]}: {obs_tuple[1]}"
            else:
                cell_title_part = ""
            title = f"'{data_annot_name}' in {data_name} vs. '{ref_annot_name}' in {ref_name}{cell_title_part}"

        dge_res = differential_gene_expression(
            data=data,
            ref_data=ref_data,
            data_annotation_tuple=data_annotation_tuple,
            ref_annotation_tuple=ref_annotation_tuple,
            obs_tuple=obs_tuple,
            region_tuple=region_tuple,
            plot_volcano=plot_volcano,
            method=method,
            ignore_duplicate_assignments=ignore_duplicate_assignments,
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

    def load_all(self,
                 skip: Optional[str] = None,
                 ):
        for xd in self._data:
            print(xd.sample_id)
            for f in LOAD_FUNCS:
                if skip is None or skip not in f:
                    func = getattr(xd, f)
                    try:
                        func()
                    except ModalityNotFoundError as err:
                        print(err)

    def load_annotations(self):
        for xd in self._data:
            print(xd.sample_id)
            xd.load_annotations()

    def load_cells(self):
        for xd in self._data:
            print(xd.sample_id)
            xd.load_cells()

    def load_images(self,
                    names: Union[Literal["all", "nuclei"], str] = "all", # here a specific image can be chosen
                    nuclei_type: Literal["focus", "mip", ""] = "mip",
                    load_cell_segmentation_images: bool = True
                    ):

        for xd in self._data:
            print(xd.sample_id)
            xd.load_images(names=names,
                           nuclei_type=nuclei_type,
                           load_cell_segmentation_images=load_cell_segmentation_images)

    def load_regions(self):
        for xd in self._data:
            print(xd.sample_id)
            xd.load_regions()

    def load_transcripts(self,
                        transcript_filename: str = "transcripts.parquet"
                        ):
        for xd in self._data:
            print(xd.sample_id)
            xd.load_transcripts()

    def plot_umaps(self,
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
            # Assuming each dataset has an AnnData object or can be converted to one
            adata = dataset.cells["main"].matrix
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

    def query(self, criteria: dict):
        """Query the experiment based on metadata criteria.

        Args:
            criteria (dict): A dictionary where keys are column names and values are lists of categories to select.

        Returns:
            InSituExperiment: A new InSituExperiment object with the selected subset.
        """
        mask = pd.Series([True] * len(self._metadata))
        for column, values in criteria.items():
            values = convert_to_list(values)
            if column in self._metadata.columns:
                mask &= self._metadata[column].isin(values)
            else:
                raise KeyError(f"Column '{column}' not found in metadata.")

        return self[mask]

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
        #for i in range(len(metadata)):
        for dataset_path in dataset_paths:
            #dataset_path = path / f"{i}"
            dataset = insitupy.read_xenium(dataset_path)
            data.append(dataset)

        # Create a new InSituExperiment object
        experiment = cls()
        experiment._metadata = metadata
        experiment._data = data
        experiment._path = path

        return experiment

    @classmethod
    def from_config(cls, config_path: Union[str, os.PathLike, Path]):
        """
        Create an InSituExperiment object from a configuration file.

        Args:
            config_path (Union[str, os.PathLike, Path]): The path to the configuration CSV or Excel file.

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
            config = pd.read_csv(config_path)
        elif config_path.suffix in ['.xlsx', '.xls']:
            config = pd.read_excel(config_path)
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
        for _, row in config.iterrows():
            dataset_path = Path(row['directory'])

            # Check if the path is relative and if so, append the current path
            if not dataset_path.is_absolute():
                dataset_path = current_path / dataset_path

            # Check if the directory exists
            if not dataset_path.exists():
                raise FileNotFoundError(f"No such directory found: {str(dataset_path)}")

            dataset = insitupy.read_xenium(dataset_path)
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
                    data: insitupy.InSituData,
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
        for xd in self._data:
            print(xd.sample_id)
            xd.remove_history()

    def save(self):
        if self.path is None:
            print("No save path found in '.self'. First save the InSituExperiment using '.saveas()'.")
            return
        else:
            parent_path_identical = [d.path.parent == self.path for d in self.data]
            if not np.all(parent_path_identical):
                print(f"Saving process failed. Save path of some InSituData objects did not lie inside the InSituExperiment save path: {self.metadata['uid'][parent_path_identical].values}")
            else:
                for xd in self._data:
                    print(xd.sample_id)
                    xd.save()


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
        for index, dataset in enumerate(self._data):
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


    def plot_overview(self, colums_to_plot: List[str] = [], layer: str = None, index: bool = True, qc_width: float = 4.0):
        """
        Plots an overview table with metadata and quality control metrics.

        Args:
            columns_to_plot (List[str]): List of column names to include in the plot.
            layer (str, optional): The layer of the AnnData object to use for calculations. If None, the function will use the main matrix (adata.X) or the 'counts' layer if the main matrix does not contain integer counts.
            index (bool, optional): Whether to add extra index or not. Default is True.
            custom_width (float, optional): Custom width for metadata columns. Default is 1.0.
            qc_width (float, optional): Width for quality control metric columns. Default is 4.0.

        Raises:
            ImportError: If the 'plottable' framework is not installed.

        Returns:
            None: Displays a plot with the overview table.
        """
        from anndata import AnnData
        try:
            from plottable import ColumnDefinition, Table
        except ImportError:
            raise ImportError("This function requires the 'plottable' framework. Please install it with 'pip install plottable'.")

        def calculate_max_cell_widths_and_sum(df, multiplier=0.2):
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


        def custom_bar(ax: Axes, val: float, max: float, color: str = None, rect_kw: dict = {}):
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

        def calculate_metrics(adata: AnnData, layer: str = None):
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
                if not is_integer_counts(adata.X):
                    if not is_integer_counts(adata.layers["counts"]):
                        warnings.warn("No raw counts provided, metrics are set to 0.")
                        return 0, 0
                    else:
                        df_cells, _ = sc.pp.calculate_qc_metrics(adata, percent_top=None, layer="counts")
                else:
                    df_cells, _ = sc.pp.calculate_qc_metrics(adata, percent_top=None)
            else:
                if not is_integer_counts(adata.layers[layer]):
                    warnings.warn(f"No raw counts provided in layer '{layer}', metrics are set to 0.")
                    return 0, 0
                else:
                    df_cells, _ = sc.pp.calculate_qc_metrics(adata, percent_top=None, layer=layer)

            return df_cells["n_genes_by_counts"].median(), df_cells["total_counts"].median()

        # Copy the metadata, select the columns to plot, and add index if nessiccary
        df = self.metadata.copy()[colums_to_plot]
        colname_tmp = "ind_tmp"
        if not index and df.shape[1] > 0:
            # Set the first column as the index if index is False
            col_id = df.columns[0]
        else:
            # Rename the index column and reset the index
            df = df.rename_axis(colname_tmp).reset_index()
            col_id = colname_tmp

        # Calculate the maximum cell widths and the total width
        width_dict, total_width = calculate_max_cell_widths_and_sum(df)
        column_definition = []
        # Add all desired columns from metadata
        for column_name in df.columns:
            border = None
            if column_name == colname_tmp:
                if index:
                    border = "right"
                column_definition.append(ColumnDefinition(name=column_name, textprops={"ha": "center"}, width=width_dict[column_name], title="", border=border))
            else:
                column_definition.append(ColumnDefinition(name=column_name, group="metadata", textprops={"ha": "center"}, width=width_dict[column_name]))

        #Calculate predefined QC metrics
        list_gene_count = []
        list_transcript_count = []
        for _, data in self.iterdata():
            if data.cells is None:
                warnings.warn("Counts were not loaded. Loading.")
                data.load_cells()
            if data.cells is None or data.cells.key_main is None or data.cells["main"].matrix is None:
                warnings.warn("Counts are not defined or loaded.")
                list_gene_count.append(0)
                list_transcript_count.append(0)
            else:
                m_gene_counts, m_transcript_counts = calculate_metrics(data.cells["main"].matrix, layer=layer)
                list_gene_count.append(m_gene_counts)
                list_transcript_count.append(m_transcript_counts)

        df["mean_transcript_counts"] = list_transcript_count
        df["mean_gene_counts"] = list_gene_count
        max_genes = df["mean_gene_counts"].max()
        max_transcripts = df["mean_transcript_counts"].max()

        # Add all columns with QC metrics
        column_definition_bars = [
            ColumnDefinition("mean_transcript_counts", group="qc_metrics", plot_fn=custom_bar, plot_kw={"max": max_transcripts}, title="Median Transcripts per Cell", textprops={"ha": "center"}, width=qc_width, border="left"),
            ColumnDefinition("mean_gene_counts", group="qc_metrics", plot_fn=custom_bar, plot_kw={"max": max_genes}, title="Median Genes per Cell", textprops={"ha": "center"}, width=qc_width)
        ]
        # Create the plot
        fig, ax = plt.subplots(figsize=(total_width + qc_width * 2, len(df) * 0.7 + 1))
        plt.rcParams["font.family"] = ["DejaVu Sans"]
        table = Table(df, column_definitions=(column_definition + column_definition_bars), row_dividers=True,
                    footer_divider=True, ax=ax, row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
                    col_label_divider_kw={"linewidth": 1, "linestyle": "-"}, column_border_kw={"linewidth": 1, "linestyle": "-"},
                    index_col=col_id,)

        plt.show()

