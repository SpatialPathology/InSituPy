# Use a Baysor segmentation

[Baysor](https://github.com/kharchenkolab/Baysor) segments cells from the transcript positions.
`InSituPy` does not ship a dedicated Baysor reader: `MultiCellData.add_baysor()` is deprecated
since 0.12 and will be removed in 0.13. There are two ways to use a Baysor segmentation instead.

## Xenium data: import the segmentation with Xenium Ranger

For Xenium data, the recommended route is to let Xenium Ranger build a new Xenium output bundle
from the Baysor result and read that bundle as usual. This keeps cells, boundaries, transcripts
and images consistent, and the result also opens in Xenium Explorer.

1. Run Baysor on the Xenium output bundle with the legacy output style (see the
   [Baysor Xenium workflow](https://github.com/kharchenkolab/Baysor/blob/master/docs/xenium.md)).
   It writes `segmentation.csv` and `segmentation_polygons_2d.json`.
2. Import the segmentation with Xenium Ranger:

   ```bash
   xeniumranger import-segmentation \
     --id baysor_xenium \
     --xenium-bundle path/to/xenium_output \
     --transcript-assignment path/to/baysor_output/segmentation.csv \
     --viz-polygons path/to/baysor_output/segmentation_polygons_2d.json \
     --units microns
   ```

3. Read the new bundle:

   ```python
   import insitupy as ispy

   xd = ispy.io.read_xenium("baysor_xenium/outs")
   ```

To keep the original 10x segmentation as well, read both bundles and add the Baysor cells as a
second layer:

```python
xd = ispy.io.read_xenium("path/to/xenium_output")
xd_baysor = ispy.io.read_xenium("baysor_xenium/outs")
xd.cells.add_celldata(cd=xd_baysor.cells["main"], key="baysor")
```

## Other data: build a cell layer from the Baysor output

For other platforms, build a `CellData` layer from the Baysor output files yourself. The code
below reads the legacy output bundle of Baysor 0.7 or later (`segmentation_counts.loom`,
`segmentation_cell_stats.csv` and `segmentation_polygons_2d.json` written with the default
`--polygon-format FeatureCollection`). It matches polygons to cells **by cell name**, because
Baysor writes polygons only for cells with a valid 2D outline, so the two files are not
guaranteed to have the same rows in the same order.

It needs `rasterio` (`pip install rasterio`) to turn the polygons into a segmentation mask. See
{doc}`Build an InSituData object from custom data <InSituPy_build_objects_from_scratch>` for what
the mask and `seg_mask_value` mean.

```python
from pathlib import Path

import anndata as ad
import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from shapely import affinity

from insitupy.containers import BoundariesData, CellData

baysor_dir = Path("path/to/baysor_output")
pixel_size = 0.2125  # µm per pixel of the segmentation mask, e.g. that of your morphology image

# 1. counts (cells x genes) and per-cell statistics, both keyed by the Baysor cell name.
#    The loom file is HDF5 with a genes x cells matrix, so h5py can read it directly.
with h5py.File(baysor_dir / "segmentation_counts.loom", "r") as f:
    adata = ad.AnnData(
        X=f["matrix"][:].T,
        obs=pd.DataFrame(index=pd.Index(f["col_attrs/Name"].asstr()[:], name="cell")),
        var=pd.DataFrame(index=pd.Index(f["row_attrs/Name"].asstr()[:], name="gene")),
    )
stats = pd.read_csv(baysor_dir / "segmentation_cell_stats.csv").set_index("cell")
adata.obs = adata.obs.join(stats)
adata.obsm["spatial"] = adata.obs[["x", "y"]].to_numpy()  # µm

# 2. polygons, matched to the table by cell name (not by position)
polygons = gpd.read_file(baysor_dir / "segmentation_polygons_2d.json").set_index("cell")
polygons = polygons[polygons.geom_type == "Polygon"]
cells = adata.obs_names.intersection(polygons.index)
adata = adata[cells].copy()
polygons = polygons.loc[cells]

# 3. rasterize the polygons (µm) into a label mask (0 = background, 1..n = cells)
seg_mask_value = np.arange(1, len(cells) + 1)
geoms_px = [affinity.scale(g, 1 / pixel_size, 1 / pixel_size, origin=(0, 0)) for g in polygons.geometry]
_, _, maxx, maxy = gpd.GeoSeries(geoms_px).total_bounds
mask = rasterize(
    zip(geoms_px, seg_mask_value),
    out_shape=(int(np.ceil(maxy)) + 1, int(np.ceil(maxx)) + 1),
    dtype=np.int32,
)

# 4. build the layer and add it to an existing InSituData object `xd`
boundaries = BoundariesData(cell_names=adata.obs_names.astype(str), seg_mask_value=seg_mask_value)
boundaries.add_boundaries(cell_boundaries=mask, pixel_size=pixel_size)
xd.cells.add_celldata(cd=CellData(table=adata, boundaries=boundaries), key="baysor")
```

Pass `is_main=True` to `add_celldata` to make the Baysor layer the main layer. Baysor's newer
Parquet output bundle (`cells.parquet`, `cell_boundaries.parquet`, `feature_matrix.h5`) works the
same way: read the counts with `scanpy.read_10x_h5` and the polygons with `geopandas.read_parquet`.

```{note}
For transcript-based segmentation, [Proseg](https://github.com/dcjones/proseg) is supported
directly through `MultiCellData.add_proseg()`; see
{doc}`Perform segmentation with Proseg <InSituPy_add_proseg_data>`.
```
