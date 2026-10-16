"""Writers for the InSituPy 0.12 ("format 1") on-disk layout, for backward-compatibility tests.

Every file is written directly with low-level libraries (json, zarr, anndata, pyarrow,
geopandas), never through an InSituPy writer, so the fixtures stay a frozen picture of the
0.12 layout while the package's own writers move to format 2. The layout mirrors what
InSituPy 0.12 writes (checked against a ``saveas()`` output on 2026-10-15):

- ``.ispy`` with versioned ``data`` pointers, ``history``, ``uids`` and the absolute
  ``path``/``metadata_file`` keys that ``read()`` injects and every later ``save()`` writes
  back;
- ``cells/<YYMMDD-HHMMSSffffff-hex8>/`` with ``.multicelldata``, ``main/.celldata``,
  ``table.h5ad`` (CSR ``X``) and ``boundaries.zarr`` (Zarr v3, pyramids per mask layer);
- ``images/<name>.zarr`` pyramids without NGFF metadata, RGB stored channels-last (``YXS``);
- ``transcripts/transcripts.parquet/`` as a directory of dask part files;
- ``units/.multispatialunitsdata`` + ``<key>/{shapes.parquet, data.h5ad, metadata.json}``;
- ``annotations/<uid>/<key>.geojson`` and ``regions/<uid>/<key>.geojson``.

Each writer returns the values it wrote, so tests can compare what InSituPy reads back.
"""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
from anndata import AnnData
from scipy import sparse
from shapely.geometry import Polygon

V1_VARIANTS = (
    "full",
    "history",
    "pre_uids",
    "zarr_v2_images",
    "flat_units",
    "no_pointer",
    "matrix_key",
    "empty_nucleus_map",
)

VERSION = "0.12.1"
N_CELLS = 6
GENES = ["g0", "g1", "g2", "g3"]
CELL_PIXEL_SIZE = 0.5
IMAGE_PIXEL_SIZE = 0.25
N_LEVELS = 3

# save-directory names (pattern YYMMDD-HHMMSSffffff-hex8); the "history" variant commits the
# OLDER-named one, so a loader that picks the newest name instead of the pointer reads the decoy
OLD_DIR = "250101-120000000000-aaaaaaaa"
NEW_DIR = "250102-120000000000-bbbbbbbb"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=4))


def _pyramid(level0: np.ndarray) -> list[np.ndarray]:
    """Downsample by striding Y/X, like a nearest-neighbour pyramid."""
    return [level0[:: 2**k, :: 2**k] for k in range(N_LEVELS)]


def _write_pyramid_group(group, levels: list[np.ndarray], channels_last_rgb: bool = False) -> None:
    for i, arr in enumerate(levels):
        chunks = (*arr.shape[:2], 1) if channels_last_rgb else arr.shape
        group.create_array(str(i), data=arr, chunks=chunks)


# ── cells ─────────────────────────────────────────────────────────────────────


def _cell_label_mask() -> np.ndarray:
    """12x12 int64 mask, cells 1..6 on a 2x3 grid of 6x4 blocks (int64 as 0.12 allowed)."""
    mask = np.zeros((12, 12), dtype=np.int64)
    for i in range(N_CELLS):
        r, c = divmod(i, 3)
        mask[r * 6:(r + 1) * 6, c * 4:(c + 1) * 4] = i + 1
    mask[0, :] = 0  # some background
    return mask


def _nucleus_label_mask() -> np.ndarray:
    """Nucleus labels (nucleus index + 1) inside each cell block."""
    mask = np.zeros((12, 12), dtype=np.int64)
    for i in range(N_CELLS):
        r, c = divmod(i, 3)
        mask[r * 6 + 2:r * 6 + 4, c * 4 + 1:c * 4 + 3] = i + 1
    return mask


def _make_table(rng: np.random.Generator, cell_names: list[str], offset: float = 0.0) -> AnnData:
    X = rng.integers(0, 6, size=(N_CELLS, len(GENES))).astype(np.float32) + offset
    obs = pd.DataFrame(
        {
            "cluster": pd.Categorical([f"k{i % 2}" for i in range(N_CELLS)]),
            "area": rng.random(N_CELLS) * 50,
        },
        index=pd.Index(cell_names),
    )
    table = AnnData(X=sparse.csr_matrix(X), obs=obs, var=pd.DataFrame(index=pd.Index(GENES)))
    table.obsm["spatial"] = rng.random((N_CELLS, 2)) * 6
    return table


def _write_cells_dir(cells_dir: Path, table: AnnData, cell_names: list[str], *, variant: str) -> dict:
    """Write one versioned cells save directory; return the boundary values written."""
    _write_json(cells_dir / ".multicelldata",
                {"key_main": "main", "all_keys": ["main"], "version": VERSION})
    table_key = "matrix" if variant == "matrix_key" else "table"
    _write_json(cells_dir / "main" / ".celldata",
                {table_key: "table.h5ad", "boundaries": "boundaries.zarr",
                 "version": VERSION, "config": {}})
    table.write_h5ad(cells_dir / "main" / "table.h5ad", compression="gzip")

    zarr_format = 2 if variant == "zarr_v2_images" else 3
    seg_mask_value = np.arange(1, N_CELLS + 1, dtype=np.uint32)
    has_map = variant != "empty_nucleus_map"
    root = zarr.open_group(cells_dir / "main" / "boundaries.zarr", mode="w", zarr_format=zarr_format)
    root.attrs["insitupy_has_nucleus_map"] = True
    root.create_array("cell_names", data=np.array(cell_names))
    root.create_array("seg_mask_value", data=seg_mask_value)
    if has_map:
        root.create_array("nucleus_to_cell_map/nucleus_index", data=np.arange(N_CELLS, dtype=np.int64))
        root.create_array("nucleus_to_cell_map/cell_name", data=np.array(cell_names))
        root.create_array("nucleus_count", data=np.ones(N_CELLS, dtype=np.int64))
    masks = {"cells": _pyramid(_cell_label_mask()), "nuclei": _pyramid(_nucleus_label_mask())}
    for layer, levels in masks.items():
        group = root.create_group(f"masks/{layer}")
        group.attrs["pixel_size"] = CELL_PIXEL_SIZE
        _write_pyramid_group(group, levels)

    return {
        "cell_names": list(cell_names),
        "seg_mask_value": seg_mask_value,
        "masks": masks,
        "mask_pixel_size": CELL_PIXEL_SIZE,
        "nucleus_to_cell_map": {i: n for i, n in enumerate(cell_names)} if has_map else {},
    }


# ── images ────────────────────────────────────────────────────────────────────


def _write_image(path: Path, levels: list[np.ndarray], *, axes: str, rgb: bool, zarr_format: int) -> None:
    root = zarr.open_group(path, mode="w", zarr_format=zarr_format)
    _write_pyramid_group(root, levels, channels_last_rgb=rgb)
    root.attrs.update({
        "filename": None,
        "shape": list(levels[0].shape),
        "axes": axes,
        "OME": {},
        "pixel_size": IMAGE_PIXEL_SIZE,
        "rgb": rgb,
    })


# ── transcripts ───────────────────────────────────────────────────────────────


def _transcripts_frame(rng: np.random.Generator, n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "x_location": rng.random(n) * 6,
        "y_location": rng.random(n) * 6,
        "z_location": np.zeros(n),
        "feature_name": pd.Categorical([GENES[i % len(GENES)] for i in range(n)]),
        "cell_id": [f"c{i % N_CELLS}" for i in range(n)],
    })


def _write_transcript_parts(directory: Path, frames: list[pd.DataFrame]) -> None:
    """Dask-style part directory; each part carries its own locally inferred dictionary width."""
    directory.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        table = pa.Table.from_pandas(frame, preserve_index=False)
        width = pa.int8() if i % 2 == 0 else pa.int16()
        field = table.schema.get_field_index("feature_name")
        table = table.set_column(
            field, "feature_name", table.column("feature_name").cast(pa.dictionary(width, pa.string()))
        )
        pq.write_table(table, directory / f"part.{i}.parquet")


# ── units ─────────────────────────────────────────────────────────────────────


def _write_units_layer(directory: Path, rng: np.random.Generator, names: list[str]) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    shapes = gpd.GeoDataFrame(
        {"name": names,
         "geometry": [Polygon([(i, i), (i + 1, i), (i + 1, i + 1), (i, i + 1)]) for i in range(len(names))]},
        index=pd.Index(names),
    )
    shapes.to_parquet(directory / "shapes.parquet")
    X = rng.random((len(names), 2))
    AnnData(X=X, obs=pd.DataFrame(index=pd.Index(names)),
            var=pd.DataFrame(index=pd.Index(["v0", "v1"]))).write_h5ad(
        directory / "data.h5ad", compression="gzip")
    _write_json(directory / "metadata.json", {"version": VERSION, "unit_type": "unit"})
    return {"names": list(names), "X": X, "wkt": [g.wkt for g in shapes.geometry]}


# ── annotations / regions ─────────────────────────────────────────────────────


def _square(x0: float, size: float) -> list:
    return [[[x0, x0], [x0 + size, x0], [x0 + size, x0 + size], [x0, x0 + size], [x0, x0]]]


def _write_geojson(path: Path, key: str, names: list[str], x0: float) -> dict:
    features = []
    rings = []
    for i, name in enumerate(names):
        ring = _square(x0 + 3 * i, 2.0)
        rings.append(Polygon(ring[0]).wkt)
        features.append({
            "type": "Feature",
            "properties": {"id": f"{name}_{i}", "origin": "manual", "layer_type": "Shapes",
                           "classification": {"name": name, "color": "#ff0000"}},
            "geometry": {"type": "Polygon", "coordinates": ring},
        })
    _write_json(path, {
        "type": "FeatureCollection",
        "name": key,
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features,
    })
    return {"names": names, "wkt": rings}


# ── project ───────────────────────────────────────────────────────────────────


def write_v1_project(root, *, variant: str = "full", seed: int = 0, uid: str | None = None,
                     slide_id: str = "slide1", sample_id: str = "s0") -> dict:
    """Write an InSituPy 0.12 project at *root* and return the values it holds.

    Args:
        root: Project directory to create (must not exist).
        variant: One of :data:`V1_VARIANTS`:

            - ``"full"``: every modality, current 0.12 layout.
            - ``"history"``: two cells and two annotations save dirs; ``.ispy`` commits the
              OLDER-named one (clock skew, B2), the newer-named one holds decoy values.
            - ``"pre_uids"``: ``.ispy`` without a ``uids`` list (B14).
            - ``"zarr_v2_images"``: images and boundaries written as Zarr v2.
            - ``"flat_units"``: pre-multi-unit flat ``units/`` layout (read as key ``"main"``).
            - ``"no_pointer"``: ``.ispy`` without ``data`` pointers (newest-by-name fallback).
            - ``"matrix_key"``: ``.celldata`` names the table under the legacy ``matrix`` key.
            - ``"empty_nucleus_map"``: boundaries marked as having a known-empty nucleus map.
        seed: Seed for all generated values.
        uid: Experiment-slot uid stored in ``.ispy`` (``None`` for a standalone project).
        slide_id: Value of the ``slide_id`` key.
        sample_id: Value of the ``sample_id`` key.

    Returns:
        Dict of expected values: ``table`` (AnnData as written), boundary values (``cell_names``,
        ``seg_mask_value``, ``masks``, ``mask_pixel_size``, ``nucleus_to_cell_map``), ``images``
        (name -> levels) with ``image_pixel_size`` and ``image_rgb`` (name -> bool),
        ``transcripts`` (DataFrame), ``units`` (key -> values), ``annotations`` / ``regions``
        (key -> names and WKT), plus ``slide_id``, ``sample_id``, ``uid``.
    """
    if variant not in V1_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {V1_VARIANTS}")
    root = Path(root)
    root.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    cell_names = [f"{sample_id}c{i}" for i in range(N_CELLS)]

    # cells
    table = _make_table(rng, cell_names)
    expected = {"table": table, "slide_id": slide_id, "sample_id": sample_id, "uid": uid}
    expected.update(_write_cells_dir(root / "cells" / OLD_DIR, table, cell_names, variant=variant))
    data = {"cells": f"cells/{OLD_DIR}"}
    history = {"cells": [], "annotations": [], "regions": []}

    # images
    zarr_format = 2 if variant == "zarr_v2_images" else 3
    gray = _pyramid(rng.integers(0, 4000, size=(32, 32)).astype(np.uint16))
    he = _pyramid(rng.integers(0, 255, size=(32, 32, 3)).astype(np.uint8))
    _write_image(root / "images" / "gray.zarr", gray, axes="YX", rgb=False, zarr_format=zarr_format)
    _write_image(root / "images" / "he.zarr", he, axes="YXS", rgb=True, zarr_format=zarr_format)
    expected["images"] = {"gray": gray, "he": he}
    expected["image_pixel_size"] = IMAGE_PIXEL_SIZE
    expected["image_rgb"] = {"gray": False, "he": True}
    data["images"] = {"gray": "images/gray.zarr", "he": "images/he.zarr"}

    # transcripts
    frames = [_transcripts_frame(rng, 10), _transcripts_frame(rng, 12)]
    _write_transcript_parts(root / "transcripts" / "transcripts.parquet", frames)
    expected["transcripts"] = pd.concat(frames, ignore_index=True)
    data["transcripts"] = "transcripts/transcripts.parquet"

    # units
    unit_names = [f"{sample_id}u{i}" for i in range(3)]
    if variant == "flat_units":
        expected["units"] = {"main": _write_units_layer(root / "units", rng, unit_names)}
    else:
        _write_json(root / "units" / ".multispatialunitsdata",
                    {"key_main": "unit", "all_keys": ["unit"], "version": VERSION})
        expected["units"] = {"unit": _write_units_layer(root / "units" / "unit", rng, unit_names)}
    data["units"] = "units"

    # annotations / regions
    expected["annotations"] = {
        "roi": _write_geojson(root / "annotations" / OLD_DIR / "roi.geojson", "roi", ["tumor", "stroma"], 0.0)
    }
    expected["regions"] = {
        "zones": _write_geojson(root / "regions" / OLD_DIR / "zones.geojson", "zones", ["zone_a"], 1.0)
    }
    data["annotations"] = f"annotations/{OLD_DIR}"
    data["regions"] = f"regions/{OLD_DIR}"

    if variant == "history":
        # newer-named, uncommitted saves with different values: must never be loaded
        decoy = _make_table(np.random.default_rng(seed + 1000), cell_names, offset=100.0)
        decoy.obs["decoy"] = True
        _write_cells_dir(root / "cells" / NEW_DIR, decoy, cell_names, variant=variant)
        _write_geojson(root / "annotations" / NEW_DIR / "roi.geojson", "roi", ["decoy"], 5.0)
        history["cells"].append(f"cells/{NEW_DIR}")
        history["annotations"].append(f"annotations/{NEW_DIR}")

    ispy = {
        "data": data,
        "history": history,
        "uids": ["5b0c8f6e-0f43-4b8a-9d3c-3f1d5e7a9b21"],
        "method": "Xenium",
        "uid": uid,
        "slide_id": slide_id,
        "sample_id": sample_id,
        "version": VERSION,
        "method_params": {"pixel_size": IMAGE_PIXEL_SIZE},
        # B8: read() injects these and every later save() writes them back, so real 0.12
        # projects carry an absolute path from wherever they were last saved
        "path": "C:/somewhere/else/old_location",
        "metadata_file": ".ispy",
    }
    if variant == "pre_uids":
        del ispy["uids"]
    if variant == "no_pointer":
        ispy["data"] = {}
        ispy["history"] = {"cells": [], "annotations": [], "regions": []}
    _write_json(root / ".ispy", ispy)
    return expected


# ── experiment ────────────────────────────────────────────────────────────────


def write_v1_experiment(root, n: int = 2, *, csv_only: bool = False) -> dict:
    """Write an InSituPy 0.12 experiment with *n* ``full`` datasets at *root*.

    Args:
        root: Experiment directory to create (must not exist).
        n: Number of datasets (``data-000`` ...).
        csv_only: Write the pre-Parquet legacy metadata (``metadata.csv`` + dtype schema
            sidecar, with the auto-generated ``slide_id``/``sample_id`` columns that the
            reader drops) instead of ``metadata.parquet``.

    Returns:
        Dict with ``metadata`` (the DataFrame InSituPy should read back), ``filters``
        (name -> mask) and ``datasets`` (uid -> :func:`write_v1_project` result).
    """
    root = Path(root)
    root.mkdir(parents=True)
    uids = [f"uid{i:03d}" for i in range(n)]
    datasets = {}
    # written in reverse so that directory order != metadata order (pairing is by uid)
    for i in reversed(range(n)):
        datasets[uids[i]] = write_v1_project(
            root / f"data-{n - 1 - i:03d}", seed=i, uid=uids[i], sample_id=f"s{i}"
        )

    metadata = pd.DataFrame({
        "uid": uids,
        "patient": [f"{i:03d}" for i in range(n)],  # leading zeros must survive
        "group": ["a" if i % 2 == 0 else "b" for i in range(n)],
    })
    if csv_only:
        legacy = metadata.assign(slide_id="slide1", sample_id=[f"s{i}" for i in range(n)])
        legacy.to_csv(root / "metadata.csv", index=True)
        _write_json(root / "metadata.schema.json", {
            "version": 1,
            "column_dtypes": {str(c): str(t) for c, t in legacy.dtypes.items()},
        })
    else:
        metadata.to_parquet(root / "metadata.parquet", index=False)

    filters = {"only_a": [g == "a" for g in metadata["group"]]}
    _write_json(root / "colors.json", {})
    _write_json(root / "filters.json", {
        "version": 2,
        "filters": {k: {"mask": v, "note": None} for k, v in filters.items()},
        "composites": {},
    })
    return {"metadata": metadata, "filters": filters, "datasets": datasets}
