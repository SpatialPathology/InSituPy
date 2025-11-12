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

.. card:: `AnnData`-based multi-sample analysis workflows
    :link: InSituPy_Anndata_workflow
    :link-type: doc

    Many analysis tools (e.g. batch correction, niche analysis, etc.) are based on the `AnnData` format. This notebook demonstrates how data of an `InSituExperiment` object can be converted into an `AnnData`, processed using such tools and how the information can be imported back into an `InSituExperiment`.

```

```{toctree}
:hidden: false
:maxdepth: 1

InSituPy_InSituExperiment
InSituPy_split_datasets
InSituPy_Pseudobulk
InSituPy_Anndata_workflow
```