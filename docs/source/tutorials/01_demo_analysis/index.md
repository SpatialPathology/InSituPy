# Single-Sample Analysis

This section provides a comprehensive guide to analyzing individual spatial transcriptomics samples using InSituPy's **`InSituData`** class. The tutorials cover the complete workflow from data preparation through advanced spatial analysis.

<center><img src="../../_static/img/insitudata_overview.svg" width="800"/></center>

## Tutorial Organization

The tutorials are organized into three stages: **Setup & Preparation**, **Core Analysis Workflow**, and **Advanced Analysis**. We recommend starting with the core workflow (tutorials 01-03) before exploring optional advanced topics. The individual tutorials build on each other, making it necessary to run them sequentially starting with the tutorial on "Automated Image Registration".

### Setup & Preparation

Follow these tutorials to learn how to download demo datasets and register histological images to the spatial omics data.
```{eval-rst}
.. card:: Download Demo Datasets
    :link: 00_InSituPy_demo_datasets
    :link-type: doc
    :link-alt: Download demo datasets

    Download example datasets to follow along with the tutorials. Includes 10x Xenium mouse brain data and other sample datasets.

.. card:: Automated Image Registration
    :link: 01_InSituPy_demo_register_images
    :link-type: doc
    :link-alt: Automated image registration

    Register histological (H&E) or immunofluorescence images to your spatial transcriptomics data using automated alignment.
```

### Core Analysis Workflows

Follow these tutorials sequentially to learn the essential analysis steps:
```{eval-rst}
.. card:: 01: Load & Explore Data
    :link: 01_InSituPy_demo_load_explore
    :link-type: doc
    :link-alt: Load and explore spatial data

    Load your first spatial transcriptomics dataset, explore its structure, and understand the `InSituData` object.

.. card:: 02: Quality Control & Preprocessing
    :link: 02_InSituPy_demo_analyze
    :link-type: doc
    :link-alt: QC and preprocessing

    Perform quality control filtering, normalization, feature selection, dimensionality reduction (PCA, UMAP), and clustering.

```

### Advanced Analysis

These tutorials cover specialized analyses and can be explored in any order:
```{eval-rst}
.. card:: Working with Annotations & Regions
    :link: 03_InSituPy_demo_annotations
    :link-type: doc
    :link-alt: Annotations and regions

    Import spatial annotations and regions of interest from external tools (QuPath, ImageJ) or create them interactively in napari.

.. card:: Cropping & Subsetting Data
    :link: 04_InSituPy_demo_crop
    :link-type: doc
    :link-alt: Crop and subset data

    Extract regions of interest, subset data by cell type or spatial location, and create focused datasets for detailed analysis.

.. card:: Cell Type Annotation
    :link: 05_InSituPy_cell_type_annotation
    :link-type: doc
    :link-alt: Cell type annotation

    Annotate cell types using marker genes, reference-based methods, or transfer labels from single-cell RNA-seq data.

.. card:: Spatial Gene Expression Patterns
    :link: 06_InSituPy_gene_expression_along_axis
    :link-type: doc
    :link-alt: Gene expression along axis

    Analyze gene expression gradients along anatomical axes or spatial trajectories to identify location-dependent patterns.

.. card:: Differential Expression & Enrichment
    :link: 07_InSituPy_differential_gene_expression
    :link-type: doc
    :link-alt: Differential expression analysis

    Perform differential gene expression analysis between cell types or spatial regions, followed by Gene Ontology enrichment analysis.

```
```{toctree}
:hidden: true
:maxdepth: 1
:glob:

*
```