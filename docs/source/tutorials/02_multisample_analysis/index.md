# Multi-sample analysis

This set of tutorials introduces the analysis of multiple samples using **InSituPy**'s `InSituExperiment` class.

<center><img src="../../_static/img/insituexperiment_overview.svg" width="800"/></center>

<br>

```{eval-rst}
.. card:: Introduction to the `InSituExperiment` class
    :link: InSituPy_InSituExperiment
    :link-type: doc

    Notebook introducing the concepts behind the `InSituExperiment` class to analyze multiple samples.

.. card:: Split datasets per sample
    :link: InSituPy_split_datasets
    :link-type: doc

    Sometimes data from multiple samples can be present in one dataset. This notebook shows how to extract the data from each individual sample.

.. card:: Generate pseudobulk data
    :link: InSituPy_Pseudobulk
    :link-type: doc

    Notebook showing how to generate pseudobulk data from an `InSituExperiment` class.

.. card:: Extract individual images from a merged dataset
    :link: InSituPy_extract_individual_images
    :link-type: doc

    Notebook showing how to extract individual images from merged multi-sample data.

.. card:: Anndata workflow for multi-sample analysis
    :link: InSituPy_Anndata_workflow
    :link-type: doc

    Notebook introducing a multi-sample analysis workflow based on AnnData objects.

.. card:: Import workflow based on the Vannan et al. paper
    :link: InSituPy_Vannan_paper_import
    :link-type: doc

    Notebook demonstrating multi-sample data from `Vannan et al. <https://www.nature.com/articles/s41588-025-02080-x>`_ paper.
```

```{toctree}
:hidden: false
:maxdepth: 1

InSituPy_InSituExperiment
InSituPy_split_datasets
InSituPy_Pseudobulk
InSituPy_extract_individual_images
InSituPy_Anndata_workflow
InSituPy_Vannan_paper_import
```