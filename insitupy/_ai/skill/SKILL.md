---
name: insitupy
description: >-
  Write correct analysis code with InSituPy (insitupy-spatial), the Python framework for
  histology-guided, multi-sample single-cell spatial transcriptomics (10x Xenium, Visium, ...).
  Use whenever a user asks how to read, preprocess, analyze, plot, or save spatial
  transcriptomics data with insitupy, or writes code that imports insitupy.
version: "0.12.0b6"   # AUTO-STAMPED by tools/generate_skill_reference.py to the package version
---

# InSituPy

## Prefer the live MCP server if present

If the `insitupy` MCP server is available in this session, prefer its live tools
(`get_data_model`, `get_workflow_guide`, `get_public_api`, `get_function_source`,
`get_docstring`, ...) over anything written here - they resolve against the actually installed
source and never go stale. The `reference/` files below are a fallback for agents **without**
the MCP server. To recover live accuracy without it, introspect the installed package directly:

```bash
python -c "import insitupy, inspect; print(inspect.signature(insitupy.pp.normalize_and_transform))"
python -c "import insitupy; help(insitupy.pl.spatial)"
python -c "import insitupy; print(insitupy.__version__)"
```

## Bootstrap: verify the package is importable

Before writing insitupy code, check that the package is actually installed in the active
environment:

```bash
python -c "import insitupy; print(insitupy.__version__)"
```

- If that raises `ModuleNotFoundError`: insitupy is not installed in the active environment.
  Install it (`pip install insitupy-spatial`) or activate the environment where it lives. **Do
  not** proceed by guessing the API from memory - that is the most common source of broken
  output.
- If the `insitupy` MCP server is present, defer to it for API shapes (see above).

## This skill is versioned - suspect a stale skill before concluding an API doesn't exist

This skill documents insitupy-spatial **v{version}** (see the frontmatter). Whenever a
function, argument, or class the user asks about is **not** in this reference, **suspect a
stale skill before concluding it does not exist.** Check the installed version:

```bash
python -c "import insitupy; print(insitupy.__version__)"
# or: pip show insitupy-spatial
```

If the installed version is newer than this skill's stamped version, the skill is out of
date - upgrade it and re-read:

- **code agents:** `pip install -U insitupy-spatial && insitupy install-skill --force`
- **web chat (uploaded ZIP or pasted `llms.txt`):** re-download the current skill ZIP / re-fetch
  the `llms.txt` URL (both are per-release).

Then re-check; if the function is still missing, use the live MCP server or introspect the
installed package.

## What InSituPy is

InSituPy (`insitupy-spatial` on PyPI) is a Python framework for **histology-guided,
multi-sample analysis of single-cell spatial transcriptomics data**, currently focused on 10x
Genomics *Xenium In Situ* data (also supports Visium and QuPath-derived projects). It is
BSD-3-Clause licensed and requires Python >= 3.12.

## Core data model

InSituPy has a two-level hierarchy:

- **`InSituData`** - a single sample/slide. Integrates all modalities: `cells` (segmented
  cell-by-gene tables, possibly multiple segmentation layers), `images` (lazy dask arrays),
  `transcripts` (per-transcript coordinates), `annotations`, `regions`, and `units`.
- **`InSituExperiment`** - aggregates multiple `InSituData` instances with a sample-level
  `metadata` DataFrame for cross-sample analysis. Subscript access returns an `InSituData`
  (`experiment[i]`).

See `reference/data_model.md` for the full container tree (fields, relationships, on-disk
layout notes).

## The workflow arc: read -> preprocess -> tools -> plot -> save

```python
import insitupy as ispy

# 1. Read (see reference/io_formats.md)
data = ispy.io.read_xenium("path/to/xenium_output/")

# 2. Preprocess (see reference/preprocessing.md)
ispy.pp.normalize_and_transform(data)
ispy.pp.cluster_cells(data)

# 3. Analyze (see reference/tools.md, reference/result_types.md)
result = ispy.tl.dge(data, target_annotation_tuple=("tumor", "region1"))

# 4. Plot (see reference/plotting.md)
ispy.pl.spatial(data, color="leiden", image_key="DAPI")
ispy.pl.umap(data, color="leiden")

# 5. Save (see reference/storage_format.md)
data.saveas("path/to/my_project/")
```

This is the typical arc for a single sample. For multi-sample analysis, build an
`InSituExperiment` instead (see `reference/workflows.md`, section 7).

## Reference index (progressive disclosure)

Read only what the current task needs:

| File | Use this when ... |
|---|---|
| `reference/public_api.md` | orienting in the top-level namespace and submodule shorthands (`pl`, `pp`, `tl`, `io`, `im`, `utils`) |
| `reference/data_model.md` | reasoning about the container hierarchy and relationships |
| `reference/workflows.md` | writing a full step-by-step recipe (multi-sample, registration, annotations, ...) |
| `reference/io_formats.md` | reading raw platform output or importing annotations/regions |
| `reference/storage_format.md` | reasoning about the on-disk `.ispy` project layout |
| `reference/preprocessing.md` | calling `pp` functions - signatures and grouping |
| `reference/plotting.md` | calling `pl` functions - signatures grouped by category |
| `reference/tools.md` | calling `tl` functions - DGE, distance, neighbors, registration |
| `reference/result_types.md` | working with objects returned by `tl.dge()` / `tl.register_images()` |
| `reference/spatialdata.md` | converting to/from the SpatialData format |
| `reference/images.md` | image I/O and lazy-loading utilities (`im`) |
| `reference/datasets.md` | loading bundled sample datasets |
| `reference/interactive.md` | the napari-based `data.show()` interactive workflow |

## Before writing non-trivial code

Read `reference/conventions_and_pitfalls.md` first - it catalogs non-obvious behavior that
isn't visible from a single docstring or signature in isolation (naming collisions between
similarly-named parameters, silent no-ops, mutual-exclusivity constraints between arguments,
and areas of the codebase under active churn).
