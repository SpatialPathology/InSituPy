# Step-by-step analysis

This section provides a detailed, step-by-step demonstration of the preprocessing, analysis, and handling of individual samples in single-cell spatial transcriptomics data using **InSituPy**. Each part of the tutorial builds upon the previous one, and it is recommended to follow the steps sequentially from top to bottom to ensure proper functionality.

```{eval-rst}
.. card:: 01: Automated image registration
    :link: 01_InSituPy_demo_register_images
    :link-type: doc
    :link-alt: Tutorial on how to use the automated image registration pipeline in `InSituPy`.

    Tutorial showing how to use the automated image registration pipeline to register histological stainings or IF stainings performed on the same slide as the scST assay.


.. card:: 02: First analysis steps
    :link: 02_InSituPy_demo_analyze
    :link-type: doc

    Tutorial with introduction into first steps of analysis including filtering, preprocessing and dimensionality reduction.

.. card:: 03: Annotations in `InSituPy`
    :link: 03_InSituPy_demo_annotations
    :link-type: doc

    Tutorial introducing how to import annotations and regions from external sources like QuPath or add them in the napari viewer.

.. card:: 04: Crop data
    :link: 04_InSituPy_demo_crop
    :link-type: doc

    Tutorial showing how to crop data.

.. card:: 05: Cell type annotation
    :link: 05_InSituPy_cell_type_annotation
    :link-type: doc

    Tutorial showing different options to perform cell type annotation.

.. card:: 06: Explore gene expression along an axis
    :link: 06_InSituPy_gene_expression_along_axis
    :link-type: doc

    Demonstration on how to explore gene expression along an axis of the dataset.

.. card:: 07: Differential gene expression and GO term enrichment analysis
    :link: 07_InSituPy_differential_gene_expression
    :link-type: doc

    Notebook showcasing differential gene expression analysis within and across multiple samples and subsequent gene ontology term enrichment analysis.
```

```{toctree}
:hidden: false
:maxdepth: 1
:glob:

*
```