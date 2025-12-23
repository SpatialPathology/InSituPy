# Data import

These tutorials explain how to import data from different technologies and tools into `InSituPy`.

```{eval-rst}
.. card:: Build an `InSituData` object from custom data
    :link: InSituPy_build_objects_from_scratch
    :link-type: doc

    Tutorial on how to generate an `InSituData` object from scratch.

.. card:: Read data from QuPath
    :link: InSituPy_add_proseg_data
    :link-type: doc

    Tutorial showing how to export data from QuPath and read it with `read_qupath_project` or `read_qupath`.

.. card:: Perform segmentation with Proseg and add the results to `InSituData`
    :link: InSituPy_import_from_QuPath
    :link-type: doc

    Tutorial on how to improve cell segmentation with Proseg and add the data to `InSituData` for downstream analysis.

.. card:: Align Visium and Xenium data
    :link: InSituPy_visium_xenium_alignment
    :link-type: doc

    Tutorial on how to align Visium spatial transcriptomics data to Xenium data for integrating data from different spatial technologies on the same tissue sample.

```

```{toctree}
:hidden: false
:maxdepth: 1
:glob:

*
```