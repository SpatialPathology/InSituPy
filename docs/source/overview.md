# InSituPy at a glance

This page explains what InSituPy is for and how it organises your data, without assuming a
background in bioinformatics. If you have heard of spatial transcriptomics but are not sure what
you could do with InSituPy, start here.

## What problem does InSituPy solve?

Spatial transcriptomics technologies such as [_Xenium In Situ_](https://www.10xgenomics.com/platforms/xenium)
measure **which genes are active in every single cell** of a tissue section, and **where in the
tissue each cell sits**. One experiment produces many different pieces of information: microscope
images, the outline of every cell, millions of detected RNA molecules, and a large table of gene
counts.

These pieces depend on each other. Methods such as Xenium (in situ sequencing) or multiplexed
FISH first detect single RNA molecules in the tissue. A step called **segmentation** then finds
the outline of every cell in the microscope images and assigns each molecule to the cell it lies
in. Counting the molecules per cell gives the **gene count table**, one row per cell and one
column per gene, which most analyses start from.

In practice, these pieces are spread over many files and tools, and a study rarely has just one
tissue section. You might have tumor and normal tissue, several patients, or samples before and
after treatment, each with its own clinical information.

InSituPy keeps all of this together:

- **Everything from one tissue section in one place**, lined up so that images, cells, molecules
  and pathology annotations refer to the same positions in the tissue.
- **Many samples in one study**, connected to a sample table with patient or treatment
  information, so that you can compare groups instead of single slides.
- **Histology in the loop**: pathologists can draw or import annotations (for example "tumor" or
  "invasion front") and InSituPy stores them together with the other modalities.

## How InSituPy organises your data

InSituPy stores your data in a hierarchy: a study contains samples, a sample contains modalities,
and some modalities have further parts inside. Knowing these levels is the key to working with
InSituPy, because you reach every piece of data by following them from the top down.

```{figure} _static/img/insitupy_data_hierarchy.svg
:alt: The InSituPy data hierarchy. An InSituExperiment holds metadata, colors, filters and several InSituData objects in its data list. Each InSituData holds images, cells, transcripts, units, annotations and regions. Cells and units each have a table and boundaries or shapes.
:width: 100%
:align: center

The InSituPy data hierarchy. Stacked boxes mean that several entries can exist side by side:
several samples in `.data`, several images in `.images`, and several named layers of `.cells` or
`.units` (for example `main` and `proseg`). Dashed boxes are standard Python types.
```

Besides the data modalities, an `InSituExperiment` carries the sample table (`.metadata`) and the
saved colours and filters of the study.

The table lists the objects on each level and how to reach them in code.

| Level              | Object                 | Think of it as                         | You reach it with                       |
| ------------------ | ---------------------- | -------------------------------------- | --------------------------------------- |
| Study              | **`InSituExperiment`** | Your whole study                       | `exp`                                   |
| Sample             | **`InSituData`**       | One tissue section (one sample)        | `exp.data[0]`, or `xd`                  |
| Modality           | e.g. `cells`, `images` | One kind of information of that sample | `xd.cells`, `xd.images`                 |
| Cell layer         | `CellData`             | The cells from one segmentation        | `xd.cells["main"]`                      |
| Table / boundaries | `AnnData` / masks      | Gene counts / outlines of those cells  | `xd.cells.table`, `xd.cells.boundaries` |

## What is inside one sample?

Each `InSituData` is split into **modalities**, one for each kind of information. A sample does
not need to have all of them.

| Modality      | What it is, in plain words                                                                           | Example use                                                                                          |
| ------------- | ---------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `images`      | The microscope pictures of the section: nuclear stain (DAPI), immunofluorescence (IF) channels, or an H&E image aligned to the same tissue | Look at the morphology behind the gene data                                                          |
| `cells`       | For every cell: how many copies of each gene were counted, where the cell is, and its outline. Extra information such as the cell type or the cluster is stored here too. See [Inside the cells](#inside-the-cells-layers-tables-and-boundaries) below | Find cell types, compare gene activity                                                               |
| `units`       | The complement of `cells`: measurement spots or groups of cells that are not single cells, for example Visium spots or niches (e.g. vessels or peritumoral regions) | Combine spot-based and single-cell data and create pseudobulk data from niches.                      |
| `transcripts` | The position of every single detected RNA molecule and its identity.                                 | Check where a gene is expressed at the highest resolution, or run another method that assigns the molecules to cells |
| `annotations` | Shapes that a person drew or imported, for example from QuPath: a tumor area, a lymph node, single points of interest | Ask "which cells are inside the tumor?"                                                              |
| `regions`     | Larger areas of the tissue, for example the separate tissue pieces on one slide                      | Separate multiple samples that were measured together on one slide.                                  |

InSituPy uses one shared coordinate system in physical units, **micrometres (µm)**, for all
modalities. Positions of cells, molecules and annotations are stored in µm directly. Images keep
their pixels, together with the pixel size in µm, so InSituPy can place them in the same
coordinate system. Because of this, everything lines up by its real size in the tissue, no matter
which microscope or resolution an image came from.

### Inside the cells: layers, tables and boundaries

Finding the borders of cells in an image (segmentation) is not an exact science, and different
methods give different results. InSituPy therefore lets one sample hold **several cell layers**,
each coming from a different segmentation (for example the default Xenium segmentation and one
from [Proseg](https://github.com/dcjones/proseg)). One of them is the **main** layer, which is
used unless you ask for another one. This way you can compare how a result depends on the
segmentation, without creating separate copies of your data. See [Add Proseg data](tutorials/03_data_import/InSituPy_add_proseg_data.ipynb)
for how to add an alternative segmentation as a new layer.

Each cell layer has two parts that always belong together:

- **`table`**: the gene count table of the cells, stored as an
  [AnnData](https://anndata.readthedocs.io) object. Besides the counts it holds everything you
  compute per cell, such as clusters, cell types or a UMAP. Because it is a standard AnnData, all
  [scanpy](https://scanpy.readthedocs.io) tools work on it directly.
- **`boundaries`**: the outline of every cell (and of its nucleus), stored as segmentation masks
  that match the rows of the table.

When you remove cells with InSituPy's functions, for example during quality control, the
boundaries are kept in step with the table, so that the outlines shown in the viewer belong to
the cells that are left.
### Cells and units: two complementary views

`cells` and `units` are two ways to describe the same tissue. `cells` works at the level of single
cells, `units` at the level of larger measurement areas that are not single cells.

|                       | `cells`                                                    | `units`                                                                      |
| --------------------- | ---------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **What one entry is** | A single cell                                              | A spot, niche or functional tissue unit that is not a single cell            |
| **Typical source**    | Segmentation of the Xenium images (e.g. default or Proseg) | Visium spots, or niches you define (vessels, peritumoral regions)            |
| **Resolution**        | Single-cell                                                | Coarser: one entry covers an area, usually many cells                        |
| **Gene counts**       | `table` (AnnData, one row per cell)                        | `table` (AnnData, one row per unit)                                          |
| **Outlines**          | `boundaries` (segmentation masks of cell and nucleus)      | `shapes` (polygons, as a GeoDataFrame)                                       |
| **Several versions**  | Yes: named cell layers, one is `main`                      | Yes: named units, one is `main`, e.g. `"visium"`                             |
| **How to reach it**   | `xd.cells["main"].table`                                   | `xd.units["visium"].table`                                                   |
| **Typical use**       | Cell types, clustering, per-cell expression                | Spot-based analysis, pseudobulk from niches, combining with single-cell data |

## What can you do with InSituPy?

The central purpose of InSituPy is the one described above: to **keep all modalities of all
samples organised and lined up**, so that you do not have to match files, coordinates and sample
information by hand. Every task below builds on that. On top of the organisation, InSituPy helps
you to:

| I want to ...                     | How InSituPy helps                                                                                   | Tutorial                                                                                             |
| --------------------------------- | ---------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **Bring my data in**              | Read Xenium, Visium and QuPath data, add alternative segmentations such as Proseg, or build a sample from your own tables | [Data import](tutorials/03_data_import/index.md)                                                     |
| **Add a histology image**         | Automatically align an H&E or IF image of the same section to the spatial data                       | [Image registration](tutorials/01_demo_analysis/01_InSituPy_demo_register_images.ipynb)              |
| **Look at my tissue**             | Open an interactive viewer ([napari](https://napari.org)) with images, cells, genes and annotations on top of each other, or make static figures | [Plotting](tutorials/04_plotting/index.md)                                                           |
| **Mark areas of interest**        | Draw annotations in the viewer or import them from QuPath, then label every cell by the area it lies in | [Annotations and regions](tutorials/01_demo_analysis/03_InSituPy_demo_annotations.ipynb)             |
| **Focus on part of a slide**      | Cut out a region or split one slide into several samples                                             | [Cropping](tutorials/01_demo_analysis/04_InSituPy_demo_crop.ipynb)                                   |
| **Find cell types**               | Quality control, clustering and cell type annotation, using the standard single-cell tools ([scanpy](https://scanpy.readthedocs.io)) under the hood | [Single-sample analysis](tutorials/01_demo_analysis/index.md)                                        |
| **Ask spatial questions**         | How does gene activity change with distance from a tumor border? Which genes differ between cells inside and outside an annotated area? | [Expression along an axis](tutorials/01_demo_analysis/06_InSituPy_gene_expression_along_axis.ipynb), [Differential expression](tutorials/01_demo_analysis/07_InSituPy_differential_gene_expression.ipynb) |
| **Measure protein signal**        | Quantify immunofluorescence intensity per cell                                                       | [Quantify IF signal](tutorials/01_demo_analysis/09_quantify_IF_signal.ipynb)                         |
| **Compare many samples**          | Collect samples with their clinical information, select groups (for example "all treated patients"), and run the same analysis on all of them | [Multi-sample analysis](tutorials/02_multisample_analysis/index.md)                                  |
| **Save and share**                | Save a sample or a whole study as one project folder and open it again later                         | [Saving and loading](tutorials/00_io/InSituPy_save_load.md)                                          |
| **Use other tools too**           | Convert to and from [SpatialData](https://spatialdata.scverse.org), the common format of the scverse community | [SpatialData](tutorials/07_spatialdata/index.md)                                                     |
| **Get help from an AI assistant** | Install the InSituPy skill or MCP server so that assistants such as Claude or ChatGPT write correct InSituPy code | [AI integration](ai_integration.md)                                                                  |

InSituPy's focus is organising and aligning your data. For the analysis itself it currently
relies mostly on established packages such as [scanpy](https://scanpy.readthedocs.io). As
analysis strategies in the community settle, more built-in analysis methods may be added over time.

## A first look at the code

InSituPy is a Python package and works best in a [Jupyter](https://jupyter.org) notebook. A
typical start takes only a few lines:

```python
import insitupy as ispy

# read one Xenium run and save it as an InSituPy project
xd = ispy.io.read_xenium("path/to/xenium_output")
xd.saveas("path/to/my_project")

# find groups of similar cells
ispy.pp.normalize_and_transform(xd)
ispy.pp.reduce_dimensions(xd)
ispy.pp.cluster_cells(xd)

# look at the result on the tissue
xd.show()                              # interactive viewer
ispy.pl.spatial(xd, keys="leiden")     # static figure
```

### Getting to the individual pieces

You reach every piece of data by following the levels from the table above. A saved project is
read first and its modalities are loaded when you need them:

```python
xd = ispy.InSituData.read("path/to/my_project")
xd.load_all()                    # load all modalities that were saved

xd.cells.table                   # gene counts of the main cell layer (AnnData)
xd.cells.boundaries              # cell and nucleus outlines of the main layer
xd.cells.keys()                  # names of all cell layers, e.g. "main", "proseg"
xd.cells["proseg"].table         # gene counts of another layer
xd.units["visium"].table         # spot-level gene counts, if the sample has units
xd.images["nuclei"]              # the nuclear stain image
xd.transcripts                   # one row per detected RNA molecule
xd.annotations                   # shapes drawn or imported by a user
```

For a study with many samples, `exp.data[i]` gives you the `InSituData` of sample `i`, and from
there the same paths apply:

```python
exp = ispy.InSituExperiment.read("path/to/my_study")
exp.load_all()

exp.metadata                     # the sample table
adata = exp.data[0].cells.table  # AnnData of the first sample

import scanpy as sc
sc.pl.umap(adata, color="leiden")   # any scanpy function works on it
```

## Where to go next

A suggested path through the documentation:

1. [Install InSituPy](installation.md).
2. [Download the demo data](tutorials/01_demo_analysis/00_InSituPy_demo_datasets.ipynb) and follow
   the [single-sample tutorials](tutorials/01_demo_analysis/index.md) in order. They show how to
   work with one tissue section.
3. Bring in your own data with the [data import](tutorials/03_data_import/index.md) tutorials and
   explore it with the [plotting](tutorials/04_plotting/index.md) tutorials.
4. Compare several samples with the [multi-sample](tutorials/02_multisample_analysis/index.md)
   tutorials.
5. Browse [all tutorials](tutorials/index.md), or look up a function in the
   [API reference](api.md).