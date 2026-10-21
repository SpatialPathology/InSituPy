"""OME-NGFF 0.5 metadata for image and mask stores."""
import numpy as np

_OME_NGFF_VERSION = "0.5"


def ngff_axes(axes: str) -> list[dict]:
    """Return the NGFF axes list for an on-disk axes string such as ``"SYX"``."""
    out = []
    for ax in axes:
        if ax in "SC":
            out.append({"name": "c", "type": "channel"})
        elif ax in "YX":
            out.append({"name": ax.lower(), "type": "space", "unit": "micrometer"})
        elif ax == "T":
            out.append({"name": "t", "type": "time"})
        else:
            raise ValueError(f"Axis '{ax}' of '{axes}' cannot be described in OME-NGFF.")
    return out


def multiscales(name: str, axes: str, pixel_size: float, level_shapes) -> dict:
    """Return one NGFF ``multiscales`` entry.

    The scale of each level is computed from the real level shapes, so capped or odd pyramids
    stay exact. The translation is always zero.
    """
    pixel_size = float(pixel_size)
    shape0 = level_shapes[0]
    datasets = []
    for i, shape in enumerate(level_shapes):
        scale = [
            pixel_size * shape0[k] / shape[k] if ax in "YX" else 1.0
            for k, ax in enumerate(axes)
        ]
        datasets.append({
            "path": str(i),
            "coordinateTransformations": [
                {"type": "scale", "scale": scale},
                {"type": "translation", "translation": [0.0] * len(axes)},
            ],
        })
    return {"name": name, "axes": ngff_axes(axes), "datasets": datasets}


def _window(start, end, vmin, vmax) -> dict:
    return {"start": start, "end": end, "min": vmin, "max": vmax}


def _dtype_range(dtype) -> tuple:
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return int(info.min), int(info.max)
    if dtype == np.bool_:
        return 0, 1
    return 0.0, 1.0


def omero_rgb(dtype) -> dict:
    """Return the ``omero`` block for an RGB image stored as three channels."""
    vmin, vmax = _dtype_range(dtype)
    channels = [
        {"label": label, "color": color, "active": True, "window": _window(vmin, vmax, vmin, vmax)}
        for label, color in (("Red", "FF0000"), ("Green", "00FF00"), ("Blue", "0000FF"))
    ]
    return {"channels": channels}


def omero_grayscale(label: str, level, dtype) -> dict:
    """Return the ``omero`` block for a single-channel image.

    The window ends at the 99th percentile of the nonzero values of *level*, which keeps one
    bright outlier from washing out the display. min/max are the dtype range (integers) or the
    data range (floats).
    """
    level = np.asarray(level)
    vmin, vmax = _dtype_range(dtype)
    if np.issubdtype(level.dtype, np.floating):
        finite = level[np.isfinite(level)]
        if finite.size:
            vmin, vmax = float(finite.min()), float(finite.max())

    nonzero = level[level != 0]
    if np.issubdtype(level.dtype, np.floating):
        nonzero = nonzero[np.isfinite(nonzero)]
    if nonzero.size:
        end = np.percentile(nonzero, 99)
        end = float(end) if np.issubdtype(level.dtype, np.floating) else int(round(end))
    else:
        end = max(1, vmax)

    return {"channels": [{
        "label": label, "color": "FFFFFF", "active": True,
        "window": _window(0, end, vmin, vmax),
    }]}


def ome_attrs(*, multiscales: dict, omero: dict | None = None, image_label: bool = False) -> dict:
    """Return the value of the ``ome`` group attribute of an NGFF 0.5 store."""
    attrs = {"version": _OME_NGFF_VERSION, "multiscales": [multiscales]}
    if omero is not None:
        attrs["omero"] = omero
    if image_label:
        attrs["image-label"] = {"version": _OME_NGFF_VERSION}
    return attrs
