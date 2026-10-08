"""Shared builders and round-trip harness for the SpatialData conversion test suite.

Not a ``test_*`` module - not collected by pytest as tests. Imported by
``test_spatialdata_convert.py``, ``test_spatialdata_roundtrip.py``,
``test_spatialdata_concat_table.py``, and ``test_spatialdata_foreign_import.py``.

Importing this module without ``spatialdata`` installed is a clean skip.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from shapely.geometry import Point, Polygon

pytest.importorskip("spatialdata")

from spatialdata import SpatialData  # noqa: E402
from spatialdata.models import Image2DModel, Labels2DModel, ShapesModel, TableModel  # noqa: E402
from spatialdata.transformations import Identity, Scale  # noqa: E402
from xarray import DataArray  # noqa: E402

from insitupy._core.data import InSituData  # noqa: E402
from insitupy.containers import CellData  # noqa: E402
from insitupy.containers.boundaries_data import BoundariesData  # noqa: E402
from insitupy.containers.spatial_units_data import SpatialUnitsData  # noqa: E402
from insitupy.experiment.data import InSituExperiment  # noqa: E402

# ── Polygon / geometry helpers ──────────────────────────────────────────────


def poly(x=0, y=0, size=10):
    return Polygon([(x, y), (x + size, y), (x + size, y + size), (x, y + size)])


def poly_gdf(*names):
    """One Polygon row per name, with unique ids."""
    return gpd.GeoDataFrame({
        "id": [f"{n}_{i}" for i, n in enumerate(names)],
        "name": list(names),
        "geometry": [poly(i * 20) for i in range(len(names))],
        "color": ["#ff0000"] * len(names),
    })


# ── InSituData / cells / units / transcripts builders ───────────────────────


def make_insitudata(
    n_cells=10,
    n_genes=5,
    seed=0,
    sample_id="s1",
    cell_prefix="cell",
    gene_names=None,
    with_boundaries=False,
    with_nucleus_boundaries=False,
    with_second_cell_layer=False,
    with_image=False,
    contiguous_seg_mask_value=True,
    with_multinucleated_cells=False,
):
    """Minimal InSituData with an expression table and spatial coordinates.

    Opt-in flags add the richer modalities (boundaries, a second cell layer,
    an image) needed by the full multi-sample fixture without forcing every
    lightweight caller to pay for them.

    ``contiguous_seg_mask_value=False`` permutes ``seg_mask_value`` relative to
    obs order (only meaningful with ``with_boundaries=True``) - the real-data
    case (Xenium mask ids are not guaranteed 1-based/contiguous), as opposed to
    every other fixture's identity-order default.

    ``with_multinucleated_cells=True`` (requires ``with_boundaries=True`` and
    ``n_cells >= 3``) builds a real, non-1:1 ``nucleus_to_cell_map``: the first
    cell gets 2 nuclei, the second gets 1, the rest get 0 - exercising both
    "multiple nuclei per cell" and "no nuclei" in one fixture. Implies nucleus
    boundaries are present regardless of ``with_nucleus_boundaries``.
    """
    rng = np.random.default_rng(seed)
    if gene_names is None:
        gene_names = [f"gene_{j}" for j in range(n_genes)]

    obs = pd.DataFrame(index=pd.Index([f"{cell_prefix}_{i}" for i in range(n_cells)]))
    var = pd.DataFrame(index=pd.Index(gene_names))
    table = AnnData(
        X=rng.integers(0, 20, size=(n_cells, len(gene_names))).astype(float),
        obs=obs, var=var,
    )
    table.obsm["spatial"] = rng.random((n_cells, 2)) * 100

    boundaries = None
    if with_boundaries:
        cell_names = np.array(table.obs_names)
        seg_mask_value = np.arange(1, n_cells + 1)
        if not contiguous_seg_mask_value:
            seg_mask_value = rng.permutation(seg_mask_value)

        nucleus_to_cell_map = None
        nucleus_count = None
        if with_multinucleated_cells:
            nucleus_to_cell_map = {0: cell_names[0], 1: cell_names[0], 2: cell_names[1]}
            nucleus_count = np.zeros(n_cells, dtype=int)
            nucleus_count[0] = 2
            nucleus_count[1] = 1

        boundaries = BoundariesData(
            cell_names=cell_names,
            seg_mask_value=seg_mask_value,
            nucleus_to_cell_map=nucleus_to_cell_map,
            nucleus_count=nucleus_count,
        )
        mask = np.zeros((n_cells, n_cells), dtype=np.uint32)
        for i, value in enumerate(seg_mask_value):
            mask[i, i] = value

        nuc_mask = None
        if with_multinucleated_cells:
            nuc_mask = np.zeros((n_cells, n_cells), dtype=np.uint32)
            # nucleus labels 1, 2, 3 (1-indexed) at distinct positions, matching
            # nucleus_to_cell_map's keys 0, 1, 2.
            nuc_mask[0, 1] = 1
            nuc_mask[0, 2] = 2
            nuc_mask[1, 1] = 3
        elif with_nucleus_boundaries:
            nuc_mask = np.zeros((n_cells, n_cells), dtype=np.uint32)
            for i, value in enumerate(seg_mask_value):
                nuc_mask[i, i] = value
        boundaries.add_boundaries(cell_boundaries=mask, nuclei_boundaries=nuc_mask, pixel_size=1)

    celldata = CellData(table=table, boundaries=boundaries)
    xd = InSituData(
        path=None, metadata=None,
        slide_id="test", sample_id=sample_id,
        method_name="test", method_params={},
    )
    xd.cells.add_celldata(cd=celldata, key="main", is_main=True)

    if with_second_cell_layer:
        rng2 = np.random.default_rng(seed + 100)
        table2 = AnnData(
            X=rng2.integers(0, 20, size=(n_cells, len(gene_names))).astype(float),
            obs=obs.copy(), var=var.copy(),
        )
        table2.obsm["spatial"] = rng2.random((n_cells, 2)) * 100
        xd.cells.add_celldata(cd=CellData(table=table2, boundaries=None), key="secondary", is_main=False)

    if with_image:
        img = rng.integers(0, 255, size=(32, 32)).astype(np.uint16)
        xd.images.add_image(
            image=img, channel_names="dapi", axes="YX", pixel_size=0.5, verbose=False,
        )

    return xd


def make_units(names, unit_type="unit", n_vars=2, seed=0):
    """Minimal SpatialUnitsData with polygon shapes and an AnnData table."""
    rng = np.random.default_rng(seed)
    gdf = gpd.GeoDataFrame({
        "name": names,
        "geometry": [Point(i, i).buffer(0.4) for i in range(len(names))],
    })
    table = AnnData(
        X=rng.random((len(names), n_vars)),
        obs=pd.DataFrame(index=pd.Index(names, dtype=str)),
        var=pd.DataFrame(index=[f"v{i}" for i in range(n_vars)]),
    )
    return SpatialUnitsData(shapes=gdf, data=table, unit_type=unit_type)


def make_transcripts_df(n=6, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "x_location": rng.random(n) * 100,
        "y_location": rng.random(n) * 100,
        "z_location": np.zeros(n),
        "feature_name": [f"gene_{i % 3}" for i in range(n)],
        "cell_id": [f"cell_{i}" for i in range(n)],
    })


# ── Multi-sample, multi-panel InSituExperiment ──────────────────────────────

# Sample 0: a "v1-like" 5-gene panel. Sample 1: a "5k-like" 8-gene panel that
# partially overlaps (3 shared genes, 5 unique) - a real presence-record case,
# unlike a fully disjoint or fully identical panel.
_PANELS = [
    [f"gene_{j}" for j in range(5)],
    [f"gene_{j}" for j in range(2, 10)],
]


def make_experiment(n_samples=2, n_cells=6):
    """Multi-sample InSituExperiment exercising every round-trippable modality.

    Each sample carries: cells (2 layers, main with boundaries), a spatial
    units layer, transcripts, one image, one annotation, and one region.
    Gene panels are heterogeneous across samples (see ``_PANELS``).
    """
    exp = InSituExperiment()

    for i in range(n_samples):
        panel = _PANELS[i % len(_PANELS)]
        xd = make_insitudata(
            n_cells=n_cells, seed=i, sample_id=f"s{i}", cell_prefix=f"s{i}cell",
            gene_names=panel, with_boundaries=True, with_second_cell_layer=True,
            with_image=True,
        )
        xd.transcripts = make_transcripts_df(seed=i)
        xd.add_units(make_units([f"s{i}u0", f"s{i}u1"], unit_type="unit", seed=i))
        xd.annotations.add_data(data=poly_gdf(f"ann{i}"), key="roi", scale_factor=1.0)
        xd.regions.add_data(data=poly_gdf(f"reg{i}"), key="roi", scale_factor=1.0)
        exp._data.append(xd)

    exp._metadata = pd.DataFrame({
        "uid": [f"sample_{i}" for i in range(n_samples)],
        "slide_id": ["slide1"] * n_samples,
        "sample_id": [f"s{i}" for i in range(n_samples)],
    })
    return exp


# ── Foreign, labels-native SpatialData (WP4) ────────────────────────────────


def make_foreign_labels_native_sdata(
    with_nucleus=True, contiguous_ids=False, n_cells=6, seed=0, nucleus_pixel_size=None,
    with_nucleus_map=False, vendor_pixel_size=None, multichannel_morphology=False,
):
    """A hand-built, labels-native SpatialData object with no InSituPy dialect.

    Mimics the shape of ``spatialdata-io`` Xenium output: Identity-transformed,
    single-scale (non-pyramidal) image and label rasters, and a table whose
    ``instance_key`` column drives the real segmentation identity - deliberately
    non-contiguous / non-1-based unless ``contiguous_ids=True``. Carries no
    ``sdata.attrs["insitupy_spatialdata_dialect"]`` key.

    ``nucleus_pixel_size``, when given, gives the nucleus label element its own
    ``Scale`` transform (pixel size in µm/pixel) instead of ``Identity`` (pixel
    size 1.0) - independent-resolution cell/nucleus masks, the real-data case a
    foreign store is not guaranteed to avoid. Requires ``with_nucleus=True``.

    ``with_nucleus_map=True`` (implies ``with_nucleus=True``) adds a
    ``nucleus_boundaries`` shapes element - spatialdata-io's map source for
    SD-B2: a GeoDataFrame indexed by nucleus label (the same 1-based mask
    values used in the ``nucleus_labels`` raster) with a ``cell_id`` column
    mapping each nucleus to a table row *shifted by one* relative to the
    raster's own same-index/same-value coincidence - a non-identity structure
    matching XOA v2-v4, which mis-parents every nucleus under the old 1:1
    fallback.

    ``vendor_pixel_size``, when given, adds the two things a real
    ``spatialdata_io.xenium()`` store always carries and that the minimal /
    auto-detect import path must handle correctly (SD-B1): a ``cell_circles``
    shapes sibling with a ``Scale([1/vendor_pixel_size, 1/vendor_pixel_size])``
    transform (so ``_resolve_foreign_pixel_size`` can recover it from a
    sibling, spatialdata-io's own convention), and a micrometre-scale
    ``obsm["spatial"]`` on the table (the vendor centroids the default import
    path must keep rather than overwrite with label-derived pixel centroids).

    ``multichannel_morphology=True`` adds a 4-channel ``morphology_focus``
    image (``Identity`` transform, like the real raster) whose ``c``
    coordinate carries real Xenium-style channel names, for SD-B11.
    """
    rng = np.random.default_rng(seed)
    size = n_cells * 4

    if with_nucleus_map:
        with_nucleus = True

    if contiguous_ids:
        instance_ids = np.arange(1, n_cells + 1)
    else:
        instance_ids = rng.permutation(np.arange(2, 2 + n_cells * 7, 7))[:n_cells]

    morphology_arr = DataArray(
        np.zeros((1, size, size), dtype=np.uint16), dims=("c", "y", "x"),
    )
    images = {
        "morphology": Image2DModel.parse(
            morphology_arr, transformations={"global": Identity()},
        )
    }
    if multichannel_morphology:
        channel_names = ["DAPI", "ATP1A1/CD45/E-Cadherin", "18S", "AlphaSMA/Vimentin"]
        morphology_focus_arr = np.zeros((len(channel_names), size, size), dtype=np.uint16)
        images["morphology_focus"] = Image2DModel.parse(
            morphology_focus_arr, dims=("c", "y", "x"), c_coords=channel_names,
            transformations={"global": Identity()},
        )

    cell_mask = np.zeros((size, size), dtype=np.uint32)
    for i, value in enumerate(instance_ids):
        cell_mask[i * 4, i * 4] = value
    cell_mask_arr = DataArray(cell_mask, dims=("y", "x"))
    labels = {
        "cell_labels": Labels2DModel.parse(
            cell_mask_arr, transformations={"global": Identity()},
        )
    }

    if with_nucleus:
        nucleus_mask = np.zeros((size, size), dtype=np.uint32)
        for i, value in enumerate(instance_ids):
            nucleus_mask[i * 4 + 1, i * 4 + 1] = value
        nucleus_mask_arr = DataArray(nucleus_mask, dims=("y", "x"))
        if nucleus_pixel_size is not None:
            nucleus_transform = Scale([nucleus_pixel_size, nucleus_pixel_size], axes=("x", "y"))
        else:
            nucleus_transform = Identity()
        labels["nucleus_labels"] = Labels2DModel.parse(
            nucleus_mask_arr, transformations={"global": nucleus_transform},
        )

    obs = pd.DataFrame({
        "cell_id": instance_ids,
        "region": pd.Categorical(["cell_labels"] * n_cells),
    })
    adata = AnnData(
        X=rng.random((n_cells, 3)),
        obs=obs,
        var=pd.DataFrame(index=[f"gene_{j}" for j in range(3)]),
    )

    if vendor_pixel_size is not None:
        # Row i's pixel centroid is (i*4, i*4) - see cell_mask above - regardless of the
        # permuted instance_id value at that row; the vendor obsm is that same physical
        # location, just expressed in micrometres instead of pixels.
        pixel_centroids = np.array([[i * 4, i * 4] for i in range(n_cells)], dtype=float)
        adata.obsm["spatial"] = pixel_centroids * vendor_pixel_size

    table = TableModel.parse(adata, region="cell_labels", region_key="region", instance_key="cell_id")

    shapes = {}
    if vendor_pixel_size is not None:
        circles_gdf = gpd.GeoDataFrame({
            "geometry": [Point(i * 4, i * 4).buffer(1.0) for i in range(n_cells)],
        })
        shapes["cell_circles"] = ShapesModel.parse(
            circles_gdf,
            transformations={"global": Scale([1 / vendor_pixel_size, 1 / vendor_pixel_size], axes=("x", "y"))},
        )

    if with_nucleus_map:
        # AnnData assigns row-position string obs_names ("0", "1", ...) when obs carries no
        # explicit index (verified empirically) - the shift-by-one map below targets those.
        shifted_cell_ids = [str((i + 1) % n_cells) for i in range(n_cells)]
        nucleus_boundaries_gdf = gpd.GeoDataFrame(
            {
                "cell_id": shifted_cell_ids,
                "geometry": [Point(i * 4 + 1, i * 4 + 1).buffer(0.4) for i in range(n_cells)],
            },
            index=pd.Index(instance_ids, dtype=int),
        )
        shapes["nucleus_boundaries"] = ShapesModel.parse(nucleus_boundaries_gdf)

    return SpatialData(images=images, labels=labels, shapes=shapes, tables={"table": table})


# ── Round-trip harness ───────────────────────────────────────────────────────


def roundtrip_through_zarr(sdata, tmp_path, name="roundtrip.zarr"):
    """Write ``sdata`` to a tmp zarr store and read it back.

    The one place a real disk round trip happens; every round-trip test and
    the disk-level export tests route through this function so there is a
    single place to adapt if the spatialdata read/write API changes across
    the ``>=0.8.0,<0.9.0`` envelope.
    """
    import spatialdata

    path = tmp_path / name
    sdata.write(path)
    return spatialdata.read_zarr(path)
