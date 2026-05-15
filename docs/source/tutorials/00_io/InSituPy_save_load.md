# Saving and Loading Data

This tutorial covers how to persist and reload `InSituData` and
`InSituExperiment` objects. InSituPy supports both full project saves and
targeted partial saves that touch only a single modality — useful when you want
to update geometries or cell results without rewriting images or other heavy
data.

---

## Method overview

### `InSituData`

| Method | What is saved |
|---|---|
| `saveas(path)` | Full write to a **new** path (first-time save) |
| `save()` | Full update to the **existing** linked project |
| `save_geometries()` | Annotations and regions only |
| `save_cells()` | Cell table and boundaries only |
| `save_images()` | Images only |

### `InSituExperiment`

| Method | What is saved |
|---|---|
| `saveas(path)` | Full write to a new path (all datasets + metadata + colors + filters) |
| `save()` | Full update (all datasets + metadata + colors + filters) |
| `save_geometries()` | Geometries for all datasets |
| `save_cells()` | Cells for all datasets |
| `save_images()` | Images for all datasets |
| `save_metadata()` | Experiment metadata CSV only |
| `save_colors()` | `colors.json` only |
| `save_filters()` | `filters.json` only |

---

## `InSituData` workflows

### First-time save

Use `saveas()` when saving an object for the first time or to a new location:

```python
xd.saveas("path/to/my_project")
```

### Updating an existing project

Once a project exists on disk, use `save()` to update it in place:

```python
xd.save()                         # saves to the linked project path
xd.save(path="path/to/project")   # or specify explicitly
```

### Saving only geometries

After editing annotations or regions (e.g. via the napari viewer), save just
the geometry data without touching cells or images:

```python
xd.save_geometries()
```

### Saving only cells

After running clustering or cell-type annotation, persist the updated cell
table without rewriting images or geometries:

```python
xd.save_cells()
```

### Saving only images

After adding or modifying images (e.g. after image registration), sync images
to disk without rewriting other modalities:

```python
xd.save_images()
xd.save_images(overwrite=True)    # overwrite existing image files
```

---

## `InSituExperiment` workflows

### First-time save

```python
exp.saveas("path/to/my_experiment")
```

### Updating an existing experiment

```python
exp.save()
```

### Partial saves

The experiment-level partial save methods iterate all datasets and call the
corresponding `InSituData` method on each:

```python
exp.save_geometries()   # annotations + regions for all datasets
exp.save_cells()        # cell tables for all datasets
exp.save_images()       # images for all datasets
```

### Experiment-level metadata

Save experiment-level files independently when only those have changed:

```python
exp.save_metadata()     # metadata.csv
exp.save_colors()       # colors.json
exp.save_filters()      # filters.json
```

---

## Loading and reloading

### Reading a saved project

```python
from insitupy import InSituData, InSituExperiment

xd  = InSituData.read("path/to/my_project")
exp = InSituExperiment.read("path/to/my_experiment")
```

### Reloading individual modalities

If you have modified data on disk (e.g. by running a partial save from another
session), reload specific modalities without re-reading the full project:

```python
xd.load_annotations()
xd.load_regions()
xd.load_cells()
```

Use `reload()` to refresh all modalities at once:

```python
xd.reload()
```

---

## When to use partial saves

| Situation | Recommended method |
|---|---|
| First time writing the project | `saveas()` |
| Ran clustering, want to persist obs columns | `save_cells()` |
| Drew or edited annotations in napari | `save_geometries()` |
| Registered a new image | `save_images()` |
| Updated experiment metadata (e.g. added a column) | `exp.save_metadata()` |
| Changed color mapping | `exp.save_colors()` |
| Everything changed | `save()` |
