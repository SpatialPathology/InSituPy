"""
MCP server for exploring and understanding the InSituPy API and codebase.

Provides both generic introspection tools (search, read source, list modules)
and InSituPy-specific tools that reflect the package's architecture.

Usage:
    python -m tools.mcp_server.server
    # or
    python tools/mcp_server/server.py
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import re
import textwrap
from pathlib import Path

from mcp.server.fastmcp import FastMCP

import insitupy as _ispy

# ---------------------------------------------------------------------------
# Bootstrap: locate the insitupy package and repo root
# ---------------------------------------------------------------------------


_PACKAGE_DIR = Path(_ispy.__file__).resolve().parent  # .../insitupy/
_REPO_ROOT = _PACKAGE_DIR.parent  # one level up
_TESTS_DIR = _REPO_ROOT / "tests"
_SEARCH_ROOTS = [_PACKAGE_DIR]
if _TESTS_DIR.exists():
    _SEARCH_ROOTS.append(_TESTS_DIR)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAX_OUTPUT = 2000  # character limit for source / docstring outputs


def _truncate(text: str, limit: int = _MAX_OUTPUT) -> str:
    """Truncate text and append a notice if it exceeds *limit* characters."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n... [truncated — full text is {len(text)} chars]"


def _resolve_object(dotted_path: str):
    """Import and return the Python object at *dotted_path*.

    Tries progressively shorter module prefixes so that
    ``insitupy._core.data.InSituData.read`` resolves correctly.
    """
    parts = dotted_path.split(".")
    obj = None
    for i in range(len(parts), 0, -1):
        module_path = ".".join(parts[:i])
        try:
            obj = importlib.import_module(module_path)
            break
        except ImportError:
            continue
    if obj is None:
        raise ImportError(f"Cannot import any prefix of '{dotted_path}'")
    for attr_name in parts[i:]:
        obj = getattr(obj, attr_name)
    return obj


def _short_doc(obj) -> str:
    """Return the first non-empty line of an object's docstring."""
    doc = inspect.getdoc(obj)
    if not doc:
        return ""
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _returns_doc(obj) -> str:
    """Extract the Returns: section from a callable's docstring, if present."""
    doc = inspect.getdoc(obj)
    if not doc:
        return ""
    in_returns = False
    result: list[str] = []
    for line in doc.splitlines():
        if re.match(r"^[Rr]eturns?:\s*$", line):
            in_returns = True
            continue
        if in_returns:
            # A non-indented line ending with ":" signals a new section header
            if line and not line[0].isspace() and line.rstrip().endswith(":"):
                break
            stripped = line.strip()
            if stripped:
                result.append(stripped)
    return " ".join(result).strip()


def _format_signature(obj) -> str:
    """Return a string representation of a callable's signature."""
    try:
        sig = inspect.signature(obj)
        return str(sig)
    except (ValueError, TypeError):
        return "(...)"


def _is_public(name: str) -> bool:
    return not name.startswith("_")


def _safe_get_members(obj, predicate=None):
    """Like inspect.getmembers but swallows errors from lazy imports."""
    members = []
    for name in dir(obj):
        if name.startswith("__"):
            continue
        try:
            value = getattr(obj, name)
            if predicate is None or predicate(value):
                members.append((name, value))
        except Exception:
            continue
    return members


# Submodules to search when a top-level __all__ name is not importable directly.
# NOTE: The preferred long-term fix is to add missing imports to insitupy/__init__.py
# so that __all__ is consistent with the top-level namespace.
_ALL_SUBMODULE_SEARCH_PATHS = [
    "insitupy.containers.dataclasses",
    "insitupy.palettes",
    "insitupy.io.data",
    "insitupy.tools.dge",
    "insitupy.tools.distance",
    "insitupy.tools.registration",
]


def _resolve_all_path(name: str) -> str | None:
    """Try to find the dotted path of *name* in known insitupy submodules.

    Returns a fully qualified path string, or None if not found.
    """
    for mod_path in _ALL_SUBMODULE_SEARCH_PATHS:
        try:
            mod = importlib.import_module(mod_path)
            if getattr(mod, name, None) is not None:
                return f"{mod_path}.{name}"
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "InSituPy",
    instructions=(
        "MCP server for exploring the InSituPy spatial transcriptomics package. "
        "Use the generic introspection tools to browse modules, classes, and source code. "
        "Use the InSituPy-specific tools for curated overviews of the data model, "
        "I/O formats, plotting API, and typical workflows."
    ),
)


# ===========================
# Generic Introspection Tools
# ===========================


@mcp.tool()
def list_modules(subpackage: str | None = None) -> str:
    """List all submodules and subpackages of insitupy.

    Args:
        subpackage: Dotted subpackage name relative to insitupy
                    (e.g. "plotting" or "utils"). If omitted, lists
                    the top-level contents of insitupy.

    Returns:
        A formatted list of module names with one-line descriptions.
    """
    if subpackage:
        full_path = f"insitupy.{subpackage}"
    else:
        full_path = "insitupy"

    try:
        pkg = importlib.import_module(full_path)
    except ImportError:
        return f"Error: Cannot import '{full_path}'. Check the subpackage name."

    pkg_path = getattr(pkg, "__path__", None)
    lines: list[str] = []

    if pkg_path is not None:
        # It's a package — iterate submodules
        for importer, modname, ispkg in pkgutil.iter_modules(pkg_path):
            child_full = f"{full_path}.{modname}"
            try:
                child = importlib.import_module(child_full)
                desc = _short_doc(child)
            except Exception:
                desc = "(could not import)"
            kind = "pkg" if ispkg else "mod"
            lines.append(f"  {modname}  [{kind}]  — {desc}" if desc else f"  {modname}  [{kind}]")
    else:
        # It's a plain module — list its public names
        for name, obj in _safe_get_members(pkg):
            if _is_public(name):
                kind = type(obj).__name__
                desc = _short_doc(obj) if callable(obj) else ""
                lines.append(f"  {name}  ({kind})  — {desc}" if desc else f"  {name}  ({kind})")

    header = f"Contents of {full_path} ({len(lines)} items):\n"
    return _truncate(header + "\n".join(lines))


@mcp.tool()
def list_classes(module_path: str) -> str:
    """List all classes defined in a given module.

    Args:
        module_path: Fully qualified module path (e.g. "insitupy.containers.dataclasses")
                     or relative to insitupy (e.g. "dataclasses.dataclasses").

    Returns:
        A list of classes with their base classes and one-line descriptions.
        For full details including constructor signature, properties, and
        methods, call get_class_info("module_path.ClassName").
    """
    if not module_path.startswith("insitupy"):
        module_path = f"insitupy.{module_path}"

    try:
        mod = importlib.import_module(module_path)
    except ImportError:
        return f"Error: Cannot import '{module_path}'."

    lines: list[str] = []
    for name, obj in _safe_get_members(mod, inspect.isclass):
        if obj.__module__ != mod.__name__:
            continue  # skip re-exports
        bases = ", ".join(b.__name__ for b in obj.__bases__ if b is not object)
        base_str = f"({bases})" if bases else ""
        desc = _short_doc(obj)
        lines.append(f"  {name}{base_str}  — {desc}" if desc else f"  {name}{base_str}")

    if not lines:
        return f"No classes defined in {module_path}."
    header = f"Classes in {module_path} ({len(lines)}):\n"
    return _truncate(header + "\n".join(lines))


@mcp.tool()
def list_functions(module_path: str) -> str:
    """List all public functions in a given module with their signatures.

    Args:
        module_path: Fully qualified module path (e.g. "insitupy.preprocessing.anndata")
                     or relative to insitupy (e.g. "preprocessing.anndata").

    Returns:
        A list of function names, signatures, and one-line descriptions.
        For full parameter types, return values, and examples, call
        get_docstring("module_path.function_name").
    """
    if not module_path.startswith("insitupy"):
        module_path = f"insitupy.{module_path}"

    try:
        mod = importlib.import_module(module_path)
    except ImportError:
        return f"Error: Cannot import '{module_path}'."

    lines: list[str] = []
    for name, obj in _safe_get_members(mod, inspect.isfunction):
        if not _is_public(name):
            continue
        if obj.__module__ != mod.__name__:
            continue  # skip re-exports
        sig = _format_signature(obj)
        desc = _short_doc(obj)
        lines.append(f"  {name}{sig}  — {desc}" if desc else f"  {name}{sig}")

    if not lines:
        return f"No public functions in {module_path}."
    header = f"Functions in {module_path} ({len(lines)}):\n"
    return _truncate(header + "\n".join(lines))


@mcp.tool()
def get_class_info(class_path: str) -> str:
    """Get detailed information about a class.

    Args:
        class_path: Dotted path to the class (e.g. "insitupy._core.data.InSituData"
                    or "_core.data.InSituData").

    Returns:
        Docstring, __init__ signature, public methods with signatures,
        properties, and base classes.
    """
    if not class_path.startswith("insitupy"):
        class_path = f"insitupy.{class_path}"

    try:
        cls = _resolve_object(class_path)
    except (ImportError, AttributeError) as exc:
        return f"Error: {exc}"

    if not inspect.isclass(cls):
        return f"Error: '{class_path}' is not a class."

    parts: list[str] = []

    # Header
    bases = ", ".join(b.__name__ for b in cls.__bases__ if b is not object)
    parts.append(f"class {cls.__name__}({bases})" if bases else f"class {cls.__name__}")
    parts.append(f"Defined in: {cls.__module__}\n")

    # Docstring
    doc = inspect.getdoc(cls)
    if doc:
        parts.append("Docstring:")
        parts.append(_truncate(textwrap.indent(doc, "  "), 800))
        parts.append("")

    # __init__
    init = getattr(cls, "__init__", None)
    if init and init is not object.__init__:
        parts.append(f"__init__{_format_signature(init)}\n")

    # Properties
    props = []
    for name in sorted(dir(cls)):
        if name.startswith("_"):
            continue
        try:
            attr = getattr(cls, name)
        except Exception:
            continue
        if isinstance(attr, property):
            desc = _short_doc(attr.fget) if attr.fget else ""
            props.append(f"  .{name}  — {desc}" if desc else f"  .{name}")
    if props:
        parts.append(f"Properties ({len(props)}):")
        parts.extend(props)
        parts.append("")

    # Methods
    methods = []
    for name in sorted(dir(cls)):
        if name.startswith("_") and name != "__getitem__":
            continue
        try:
            attr = getattr(cls, name)
        except Exception:
            continue
        if callable(attr) and not isinstance(attr, property):
            sig = _format_signature(attr)
            desc = _short_doc(attr)
            is_cls = isinstance(inspect.getattr_static(cls, name), classmethod)
            prefix = "@classmethod " if is_cls else ""
            methods.append(
                f"  {prefix}{name}{sig}  — {desc}" if desc else f"  {prefix}{name}{sig}"
            )
    if methods:
        parts.append(f"Methods ({len(methods)}):")
        parts.extend(methods)

    return _truncate("\n".join(parts))


@mcp.tool()
def get_function_source(dotted_path: str) -> str:
    """Get the source code of a function, method, or class.

    Args:
        dotted_path: Dotted path (e.g. "insitupy._core.data.InSituData.crop"
                     or "_core.data.InSituData.crop").

    Returns:
        The source code (truncated to 2000 characters if needed).
    """
    if not dotted_path.startswith("insitupy"):
        dotted_path = f"insitupy.{dotted_path}"

    try:
        obj = _resolve_object(dotted_path)
    except (ImportError, AttributeError) as exc:
        return f"Error: {exc}"

    try:
        source = inspect.getsource(obj)
    except (OSError, TypeError):
        return f"Error: Source code not available for '{dotted_path}'."

    return _truncate(source)


@mcp.tool()
def get_docstring(dotted_path: str) -> str:
    """Get the full docstring of a module, class, or function.

    Args:
        dotted_path: Dotted path (e.g. "insitupy.plotting" or
                     "insitupy._core.data.InSituData.crop").

    Returns:
        The full docstring text.
    """
    if not dotted_path.startswith("insitupy"):
        dotted_path = f"insitupy.{dotted_path}"

    try:
        obj = _resolve_object(dotted_path)
    except (ImportError, AttributeError) as exc:
        return f"Error: {exc}"

    doc = inspect.getdoc(obj)
    if not doc:
        return f"No docstring found for '{dotted_path}'."
    return _truncate(doc)


@mcp.tool()
def search_codebase(pattern: str, file_glob: str = "*.py", max_results: int = 30) -> str:
    """Search across all Python files in the InSituPy source tree.

    Searches both the ``insitupy/`` package directory and ``tests/`` (if present).

    Args:
        pattern: Regex pattern to search for in file contents.
        file_glob: Glob pattern to filter files (default: "*.py").
        max_results: Maximum number of matching lines to return (default: 30).

    Returns:
        Matching file paths with line numbers and content.
    """
    regex = re.compile(pattern, re.IGNORECASE)
    results: list[str] = []
    count = 0

    all_files: list[Path] = []
    for root in _SEARCH_ROOTS:
        all_files.extend(root.rglob(file_glob))
    all_files.sort()

    for py_file in all_files:
        try:
            text = py_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        rel = py_file.relative_to(_REPO_ROOT)
        for lineno, line in enumerate(text.splitlines(), 1):
            if regex.search(line):
                results.append(f"  {rel}:{lineno}  {line.rstrip()}")
                count += 1
                if count >= max_results:
                    results.append(f"\n... (stopped at {max_results} results)")
                    return "\n".join(results)

    if not results:
        return f"No matches for pattern '{pattern}' in {file_glob} files."
    return "\n".join(results)


@mcp.tool()
def list_test_files() -> str:
    """List all test files in the tests/ directory.

    Returns:
        File names with line counts and a one-line description extracted
        from the module docstring (if present).
    """
    if not _TESTS_DIR.exists():
        return "tests/ directory not found in the repository."

    test_files = sorted(
        f for f in _TESTS_DIR.rglob("*.py")
        if f.stem.startswith("test_") or f.stem.endswith("_test")
    )

    if not test_files:
        return "No test files found in tests/."

    lines_out: list[str] = []
    for tf in test_files:
        try:
            source = tf.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        source_lines = source.splitlines()
        line_count = len(source_lines)
        # Extract first non-empty line of the module docstring
        description = ""
        try:
            tree = ast.parse(source)
            raw_doc = ast.get_docstring(tree)
            if raw_doc:
                for doc_line in raw_doc.splitlines():
                    doc_line = doc_line.strip()
                    if doc_line:
                        description = doc_line
                        break
        except SyntaxError:
            pass
        rel = tf.relative_to(_REPO_ROOT)
        suffix = f"  — {description}" if description else ""
        lines_out.append(f"  {rel}  ({line_count} lines){suffix}")

    return "\n".join(lines_out)


@mcp.tool()
def read_source_file(
    file_path: str,
    start_line: int = 1,
    end_line: int | None = None,
) -> str:
    """Read a source file from the InSituPy repository.

    Args:
        file_path: Path relative to the repo root (e.g. "insitupy/_core/data.py")
                   or a dotted module path (e.g. "insitupy._core.data").
        start_line: First line to include (1-based, default: 1).
        end_line: Last line to include (default: start_line + 200).

    Returns:
        The file contents with line numbers.
    """
    # Resolve dotted module path to file
    if not file_path.endswith(".py") and "/" not in file_path and "\\" not in file_path:
        candidate = _REPO_ROOT / Path(file_path.replace(".", "/") + ".py")
        if not candidate.exists():
            candidate = _REPO_ROOT / Path(file_path.replace(".", "/")) / "__init__.py"
        resolved = candidate
    else:
        resolved = _REPO_ROOT / Path(file_path)

    resolved = resolved.resolve()

    # Security: ensure the file is within the repo
    try:
        resolved.relative_to(_REPO_ROOT)
    except ValueError:
        return f"Error: Path '{file_path}' is outside the repository."

    if not resolved.exists():
        return f"Error: File not found: {file_path}"

    try:
        text = resolved.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"Error reading file: {exc}"

    lines = text.splitlines()
    if end_line is None:
        end_line = min(start_line + 200, len(lines))

    start_line = max(1, start_line)
    end_line = min(end_line, len(lines))

    selected = lines[start_line - 1 : end_line]
    numbered = [f"{i:>5}  {line}" for i, line in enumerate(selected, start_line)]
    header = f"File: {resolved.relative_to(_REPO_ROOT)} (lines {start_line}-{end_line} of {len(lines)})\n"
    return _truncate(header + "\n".join(numbered), 4000)


# ============================
# InSituPy-Specific Tools
# ============================


@mcp.tool()
def get_data_model() -> str:
    """Get a structured overview of the InSituPy data model.

    Returns:
        The hierarchy of core classes, their relationships, key fields,
        and how they compose into InSituData and InSituExperiment.
    """
    return textwrap.dedent("""\
    # InSituPy Data Model

    ## Core Containers

    InSituExperiment
    ├── data: list[InSituData]          # Multiple samples/slides
    ├── metadata: pd.DataFrame          # Sample-level metadata (slide_id, sample_id, custom cols)
    ├── colors: dict                    # Color schemes for visualization
    └── filters: FilterManager          # Filtering subsystem

    InSituData
    ├── metadata: dict                  # Experiment metadata, history, method info
    ├── slide_id / sample_id: str       # Identifiers
    ├── images: ImageData               # Image modalities (DAPI, IF, H&E, ...)
    ├── cells: MultiCellData            # Cell segmentations (supports multiple)
    ├── transcripts: DataFrame          # Per-transcript coordinates (dask or pandas)
    ├── annotations: AnnotationsData    # User-drawn geometric annotations
    ├── regions: RegionsData            # Tissue region polygons
    └── units: SpatialUnitsData         # Spatial units (Visium spots, niches)

    ## Data Classes

    MultiCellData
    ├── layers: dict[str, CellData]     # Named cell segmentation layers
    └── main_key: str                   # Which layer is the "primary" one

    CellData
    ├── table: AnnData                  # cells × genes matrix + obs/var/obsm
    │   ├── .X                          # Expression matrix
    │   ├── .obs                        # Cell metadata (area, cluster, ...)
    │   ├── .obsm["spatial"]            # (x, y) coordinates
    │   └── .var                        # Gene metadata
    └── boundaries: BoundariesData      # Segmentation masks

    BoundariesData
    ├── cell_names: dask.Array          # Cell identifiers
    ├── seg_mask_value: dask.Array      # Mask value → cell mapping
    ├── nucleus_to_cell_map: dict       # For multinucleated cells (Xenium v2+)
    └── nucleus_count: np.ndarray       # Nuclei per cell (optional)

    ImageData
    ├── Internal dict: name → dask.Array  # Lazy-loaded image arrays
    └── metadata: dict                    # Per-image: shape, axes, pixel_size, OME

    AnnotationsData / RegionsData
    ├── Internal dict: key → GeoDataFrame   # Polygon/point geometries
    └── metadata: dict                       # Per-key metadata

    SpatialUnitsData
    ├── shapes: GeoDataFrame            # Unit geometries
    ├── table: AnnData                  # Associated omics data
    └── unit_type: str                  # E.g. "Visium_spot", "niche"

    ## Key Relationships
    - InSituExperiment[i] → InSituData (subscript access)
    - InSituData.cells.table → AnnData (scanpy-compatible)
    - InSituData.cells.boundaries → segmentation masks as dask arrays
    - Annotations/Regions are GeoDataFrames assignable to cells via .assign_annotations()
    - Images are dask arrays, supporting lazy loading and zarr-backed pyramids
    """)


@mcp.tool()
def get_io_formats() -> str:
    """List all supported I/O formats and reader functions.

    Returns:
        Supported formats with their reader functions, required input files,
        and output types.
    """
    return textwrap.dedent("""\
    # InSituPy I/O Formats

    ## Reading Raw Data (insitupy.io)

    | Function            | Format     | Key Input Files                          | Output       |
    |---------------------|------------|------------------------------------------|--------------|
    | read_xenium()       | 10x Xenium | cell_feature_matrix.h5, cells.parquet,   | InSituData   |
    |                     |            | cells.zarr.zip, morphology images        |              |
    | read_visium()       | 10x Visium | filtered_feature_bc_matrix.h5,           | InSituData   |
    |                     |            | spatial/tissue_positions.csv, images     |              |
    | read_qupath()       | QuPath     | QuPath project directory                 | InSituData   |
    | read_qupath_project() | QuPath   | QuPath project file                      | InSituData   |
    | read_any()          | Generic    | Varies                                   | InSituData   |

    ## InSituPy Native Format

    | Function             | Direction | Format                                   |
    |----------------------|-----------|------------------------------------------|
    | InSituData.saveas()  | Write     | InSituPy project folder (.ispy metadata) |
    | InSituData.save()    | Write     | Update existing project                  |
    | InSituData.read()    | Read      | InSituPy project folder                  |
    | InSituExperiment.saveas() | Write | Experiment folder with sub-projects     |

    ## Annotation / Region Import

    | Method                        | Supported Formats                        |
    |-------------------------------|------------------------------------------|
    | InSituData.import_annotations() | GeoJSON, Shapefile, QuPath GeoJSON     |
    | InSituData.import_regions()     | GeoJSON, Shapefile, QuPath GeoJSON     |

    ## Alternative Segmentation Import

    | Method                         | Source                                   |
    |--------------------------------|------------------------------------------|
    | MultiCellData.add_baysor()     | Baysor output (segmentation_polygons.json) |
    | MultiCellData.add_proseg()     | Proseg output (transcript assignments)    |

    ## SpatialData Integration (insitupy.spatialdata)

    | Function                   | Direction | Description                         |
    |----------------------------|-----------|-------------------------------------|
    | convert_to_spatialdata()   | Export    | InSituData → SpatialData object     |
    | convert_from_spatialdata() | Import    | SpatialData → InSituData            |

    ## On-Disk Storage Structure

    project_folder/
    ├── .ispy                    # JSON metadata (slide_id, paths, history)
    ├── cells/                   # MultiCellData layers
    │   └── <timestamp_uid>/
    │       ├── .celldata        # Layer metadata
    │       ├── table.h5ad       # AnnData (scanpy format)
    │       └── boundaries.zarr.zip  # Segmentation masks (zarr)
    ├── images/                  # ImageData
    │   └── <name>.zarr/         # Pyramidal zarr arrays
    ├── transcripts/
    │   └── transcripts.parquet  # Per-transcript coordinates
    ├── units/                   # SpatialUnitsData
    │   ├── shapes.parquet
    │   ├── data.h5ad
    │   └── metadata.json
    ├── annotations/             # AnnotationsData
    │   └── <timestamp_uid>/
    │       └── <key>.geojson
    └── regions/                 # RegionsData
        └── <timestamp_uid>/
            └── <key>.geojson
    """)


@mcp.tool()
def get_plotting_api() -> str:
    """List all plotting functions grouped by category.

    Returns:
        Plotting functions with their key parameters and descriptions.
    """
    lines: list[str] = ["# InSituPy Plotting API (insitupy.pl)\n"]

    # Try to get live info from the plotting module
    try:
        import insitupy.plotting as pl_mod

        categories = {
            "Spatial": ["spatial", "plot_spatial"],
            "Embedding / UMAP": ["embedding", "umap"],
            "Overview": ["overview", "plot_overview"],
            "Cell Composition": ["cellular_composition", "plot_cellular_composition"],
            "QC": ["plot_qc_metrics"],
            "DGE / Volcano": ["volcano", "dual_foldchange_plot"],
            "Expression Along Axis": ["cell_abundance_along_axis", "cell_expression_along_axis"],
            "Utility": ["colorlegend", "plot_colorlegend", "test_transformations"],
            "Configuration": ["DataConfig", "LayoutConfig", "PlotConfig"],
        }

        for cat_name, func_names in categories.items():
            lines.append(f"## {cat_name}")
            for fname in func_names:
                obj = getattr(pl_mod, fname, None)
                if obj is None:
                    continue
                if inspect.isclass(obj):
                    lines.append(f"  {fname}  (config class)  — {_short_doc(obj)}")
                else:
                    sig = _format_signature(obj)
                    desc = _short_doc(obj)
                    lines.append(f"  {fname}{sig}")
                    if desc:
                        lines.append(f"    {desc}")
            lines.append("")
        # pca / tsne — in insitupy.plotting.scatter but not re-exported at pl top-level
        try:
            _scatter_mod = importlib.import_module("insitupy.plotting.scatter")
            pca_tsne_added = False
            for fname in ("pca", "tsne"):
                obj = getattr(_scatter_mod, fname, None)
                if obj is not None:
                    if not pca_tsne_added:
                        lines.append("## Embedding / UMAP (scatter submodule)")
                        pca_tsne_added = True
                    sig = _format_signature(obj)
                    desc = _short_doc(obj)
                    lines.append(f"  {fname}{sig}")
                    if desc:
                        lines.append(f"    {desc}")
            if pca_tsne_added:
                lines.append("")
        except Exception:
            pass

        # FACS — lives in its own submodule, may not be re-exported at pl top-level
        try:
            _facs_mod = importlib.import_module("insitupy.plotting.facs")
            facs_obj = getattr(_facs_mod, "facs", None)
            if facs_obj is not None:
                lines.append("## FACS")
                sig = _format_signature(facs_obj)
                desc = _short_doc(facs_obj)
                lines.append(f"  facs{sig}")
                if desc:
                    lines.append(f"    {desc}")
                lines.append("")
        except Exception:
            pass

    except Exception as exc:
        lines.append(f"(Could not introspect plotting module: {exc})")

    return _truncate("\n".join(lines), 10000)


@mcp.tool()
def get_preprocessing_api() -> str:
    """List all preprocessing functions with signatures.

    Returns:
        Preprocessing functions grouped by target (AnnData vs Experiment).
    """
    lines: list[str] = ["# InSituPy Preprocessing API (insitupy.pp)\n"]

    try:
        # AnnData-level preprocessing
        import insitupy.preprocessing.anndata as pp_adata

        lines.append("## AnnData-level (insitupy.preprocessing.anndata)")
        for name, obj in _safe_get_members(pp_adata, inspect.isfunction):
            if _is_public(name) and obj.__module__ == pp_adata.__name__:
                sig = _format_signature(obj)
                desc = _short_doc(obj)
                lines.append(f"  {name}{sig}")
                if desc:
                    lines.append(f"    {desc}")
        lines.append("")
    except Exception as exc:
        lines.append(f"(Could not introspect anndata preprocessing: {exc})\n")

    try:
        # Experiment-level preprocessing
        import insitupy.preprocessing.experiment as pp_exp

        lines.append("## Experiment-level (insitupy.preprocessing.experiment)")
        for name, obj in _safe_get_members(pp_exp, inspect.isfunction):
            if _is_public(name) and obj.__module__ == pp_exp.__name__:
                sig = _format_signature(obj)
                desc = _short_doc(obj)
                lines.append(f"  {name}{sig}")
                if desc:
                    lines.append(f"    {desc}")
        lines.append("")
    except Exception as exc:
        lines.append(f"(Could not introspect experiment preprocessing: {exc})\n")

    try:
        # Filtering
        import insitupy.preprocessing.filtering as pp_filt

        lines.append("## Filtering (insitupy.preprocessing.filtering)")
        for name, obj in _safe_get_members(pp_filt, inspect.isfunction):
            if _is_public(name) and obj.__module__ == pp_filt.__name__:
                sig = _format_signature(obj)
                desc = _short_doc(obj)
                lines.append(f"  {name}{sig}")
                if desc:
                    lines.append(f"    {desc}")
        lines.append("")
    except Exception as exc:
        lines.append(f"(Could not introspect filtering: {exc})\n")

    try:
        # Pseudobulk — use importlib to avoid the name collision in __init__.py
        # (preprocessing/__init__.py imports pseudobulk() function which shadows the module)
        pp_pseudo = importlib.import_module("insitupy.preprocessing.pseudobulk")

        lines.append("## Pseudobulk (insitupy.preprocessing.pseudobulk)")
        for name, obj in _safe_get_members(pp_pseudo, inspect.isfunction):
            if _is_public(name) and obj.__module__ == pp_pseudo.__name__:
                sig = _format_signature(obj)
                desc = _short_doc(obj)
                lines.append(f"  {name}{sig}")
                if desc:
                    lines.append(f"    {desc}")
        lines.append("")
    except Exception as exc:
        lines.append(f"(Could not introspect pseudobulk: {exc})\n")

    return _truncate("\n".join(lines), 5000)


@mcp.tool()
def get_tools_api() -> str:
    """List all analysis tools with signatures and descriptions.

    Returns:
        Analysis tools (DGE, distance, neighbors, permutation, pseudobulk,
        registration) with their signatures. Analysis functions return
        structured result objects (e.g. DiffExprResults, ImageRegistration);
        call get_result_types() for their full field and method documentation.
    """
    lines: list[str] = ["# InSituPy Analysis Tools (insitupy.tl)\n"]

    tool_modules = [
        ("Differential Gene Expression", "insitupy.tools.dge"),
        ("Distance Calculations", "insitupy.tools.distance"),
        ("Neighbor Detection", "insitupy.tools.neighbors"),
        ("Permutation Tests", "insitupy.tools.permutation"),
        ("Pseudobulk", "insitupy.tools.pseudobulk"),
        ("Image Registration", "insitupy.tools.registration"),
    ]

    for section_name, mod_path in tool_modules:
        lines.append(f"## {section_name}")
        try:
            mod = importlib.import_module(mod_path)
            for name, obj in _safe_get_members(mod):
                if not _is_public(name):
                    continue
                if inspect.isfunction(obj) and obj.__module__ == mod.__name__:
                    sig = _format_signature(obj)
                    desc = _short_doc(obj)
                    lines.append(f"  {name}{sig}")
                    if desc:
                        lines.append(f"    {desc}")
                elif inspect.isclass(obj) and obj.__module__ == mod.__name__:
                    desc = _short_doc(obj)
                    lines.append(f"  {name}  (class)  — {desc}" if desc else f"  {name}  (class)")
        except Exception as exc:
            lines.append(f"  (Could not introspect: {exc})")
        lines.append("")

    return _truncate("\n".join(lines), 3000)


@mcp.tool()
def get_public_api() -> str:
    """Get everything exported from the top-level insitupy namespace.

    Returns:
        All public names from ``import insitupy`` and the submodule
        shorthands (pl, pp, tl, io, im, utils).
    """
    lines: list[str] = ["# InSituPy Public API\n"]
    lines.append("## Top-level exports (from insitupy import ...)\n")

    all_names = getattr(_ispy, "__all__", [])
    for name in sorted(all_names):
        try:
            obj = getattr(_ispy, name)
            kind = type(obj).__name__
            desc = _short_doc(obj) if callable(obj) or inspect.isclass(obj) else ""
            lines.append(f"  {name}  ({kind})  — {desc}" if desc else f"  {name}  ({kind})")
        except Exception:
            # NOTE: preferred long-term fix is to add missing imports to
            # insitupy/__init__.py so that __all__ is consistent with the
            # top-level namespace.
            path = _resolve_all_path(name)
            if path:
                lines.append(f"  {name}  → accessible as {path}")
            else:
                lines.append(f"  {name}  (not found at top-level; __all__ may be stale)")

    lines.append("\n## Submodule shorthands\n")
    shorthands = {
        "insitupy.io": "I/O — read_xenium, read_visium, read_qupath, etc.",
        "insitupy.pl (plotting)": "Plotting — spatial, umap, volcano, overview, etc.",
        "insitupy.pp (preprocessing)": "Preprocessing — normalize, filter, pseudobulk, etc.",
        "insitupy.tl (tools)": "Analysis — DGE, distance, neighbors, registration, etc.",
        "insitupy.im (images)": "Image utilities — read, transform, etc.",
        "insitupy.utils": "Utilities — XeniumPanels, mock data, DGE helpers, etc.",
        "insitupy.datasets": "Example datasets — download and load sample data.",
        "insitupy.interactive": "Interactive napari-based visualization widgets.",
        "insitupy.spatialdata": "SpatialData conversion — to/from SpatialData format.",
    }
    for key, desc in shorthands.items():
        lines.append(f"  {key}: {desc}")

    return "\n".join(lines)


@mcp.tool()
def get_workflow_guide() -> str:
    """Get typical user workflows for InSituPy.

    Returns:
        Step-by-step workflow examples for common tasks.
    """
    return textwrap.dedent("""\
    # InSituPy Workflow Guide

    ## 1. Read Raw Xenium Data and Create a Project

    ```python
    import insitupy as ispy

    # Read from 10x Xenium output directory
    data = ispy.io.read_xenium("path/to/xenium_output/")

    # Optionally load transcripts (lazy by default)
    data.load_transcripts()

    # Save as InSituPy project for fast future loading
    data.saveas("path/to/my_project/")
    ```

    ## 2. Load an Existing Project

    ```python
    data = ispy.InSituData.read("path/to/my_project/")

    # Load specific modalities (lazy loading)
    data.load_images()
    data.load_cells()
    data.load_annotations()
    ```

    ## 3. Standard Single-Cell Analysis

    ```python
    adata = data.cells.table  # Access the AnnData object

    # Standard scanpy workflow
    import scanpy as sc
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(adata)

    # Or use InSituPy preprocessing shortcuts
    ispy.pp.normalize_and_transform(data)
    ispy.pp.cluster_cells(data)
    ```

    ## 4. Spatial Visualization

    ```python
    # Interactive napari viewer
    data.show(keys=["DAPI"], cells_layer="main")

    # Static matplotlib plots
    ispy.pl.spatial(data, color="leiden", image_key="DAPI")
    ispy.pl.umap(data, color="leiden")
    ```

    ## 5. Annotations and Regions

    ```python
    # Import annotations from GeoJSON / QuPath
    data.import_annotations(files="path/to/annotations.geojson", keys="tumor")
    data.import_regions(files="path/to/regions.geojson", keys="tissue")

    # Assign annotations to cells (spatial join)
    data.assign_annotations(keys="all")
    data.assign_regions(keys="all")
    # → Adds columns to data.cells.table.obs
    ```

    ## 6. Crop to a Region of Interest

    ```python
    cropped = data.crop(xlim=(1000, 3000), ylim=(2000, 5000))
    cropped.saveas("path/to/cropped_project/")
    ```

    ## 7. Multi-Sample Experiment

    ```python
    exp = ispy.InSituExperiment()
    exp.add("sample1/", metadata={"condition": "tumor", "patient": "P1"})
    exp.add("sample2/", metadata={"condition": "normal", "patient": "P1"})

    exp.load_cells()

    # Differential gene expression between conditions
    result = exp.dge(target_id=0, ref_id=1)

    # Combined AnnData for batch analysis
    combined = exp.to_anndata()
    ```

    ## 8. Image Registration (Multi-Modal Alignment)

    ```python
    reg = ispy.tl.register_images(
        source=data_he,
        target=data_xenium,
        source_image="H&E",
        target_image="DAPI"
    )
    # Apply transformation
    data_he.transform(reg.transformation_matrix, ...)
    ```

    ## 9. Alternative Segmentations

    ```python
    # Add Baysor segmentation as an additional layer
    data.cells.add_baysor("path/to/baysor_output/", pixel_size=0.2125)

    # Switch between segmentation layers
    data.cells.set_main("baysor")
    ```
    """)


@mcp.tool()
def get_storage_format() -> str:
    """Describe the on-disk storage architecture of InSituPy projects.

    Returns:
        Directory structure, metadata JSON schema (.ispy), zarr layout,
        parquet files, and GeoJSON conventions.
    """
    _version = getattr(_ispy, "__version__", "unknown")
    return textwrap.dedent("""\
    # InSituPy Storage Format

    ## Directory Layout

    ```
    project_folder/
    ├── .ispy                           # Project metadata (JSON)
    ├── cells/
    │   └── <timestamp>_<uid>/          # Versioned cell data
    │       ├── .celldata               # CellData metadata (JSON)
    │       ├── .multicelldata          # MultiCellData metadata (JSON, if multiple layers)
    │       ├── table.h5ad              # AnnData (cells × genes, obs, var, obsm)
    │       └── boundaries.zarr.zip     # Segmentation masks
    │           ├── cell_names/         # Zarr array of cell IDs
    │           ├── seg_mask_value/     # Zarr array mapping mask values → cells
    │           ├── data/0/             # Cell mask pyramid level 0 (full res)
    │           ├── data/1/             # Cell mask pyramid level 1 (downsampled)
    │           └── nuclei/0/           # Nucleus mask (optional)
    ├── images/
    │   └── <image_name>.zarr/          # OME-Zarr pyramidal image
    │       ├── 0/                      # Full resolution
    │       ├── 1/                      # 2× downsampled
    │       └── .zattrs                 # OME metadata, axes, pixel_size
    ├── transcripts/
    │   └── transcripts.parquet         # Columns: x, y, gene, qv, ...
    ├── units/
    │   ├── shapes.parquet              # GeoDataFrame with geometries
    │   ├── data.h5ad                   # Associated AnnData
    │   └── metadata.json               # Unit type, pixel size, etc.
    ├── annotations/
    │   └── <timestamp>_<uid>/
    │       └── <key>.geojson           # Annotation polygons/points
    └── regions/
        └── <timestamp>_<uid>/
            └── <key>.geojson           # Region polygons
    ```

    ## .ispy Metadata Schema

    ```json
    {
      "slide_id": "string",
      "sample_id": "string",
      "version": "VERSION_PLACEHOLDER",
      "method": "Xenium",
      "method_params": {
        "pixel_size": 0.2125,
        "xenium_version": "2.0"
      },
      "data": {
        "cells": "cells/<timestamp>_<uid>",
        "images": {
          "morphology_focus": "images/morphology_focus.zarr",
          "DAPI": "images/DAPI.zarr"
        },
        "transcripts": "transcripts/transcripts.parquet",
        "units": "units",
        "annotations": "annotations/<timestamp>_<uid>",
        "regions": "regions/<timestamp>_<uid>"
      },
      "history": {
        "cells": ["cells/<old_timestamp>_<old_uid>"],
        "annotations": [],
        "regions": []
      },
      "uids": ["<uid_history>"],
      "cropping_history": {
        "xlim": [[0, 5000]],
        "ylim": [[0, 5000]]
      }
    }
    ```

    ## Key Conventions
    - All spatial coordinates are in pixels at the native resolution
    - pixel_size is in µm/pixel (e.g. 0.2125 for Xenium)
    - Boundaries are stored as label masks (integer arrays where pixel value = cell ID)
    - Images use OME-Zarr with multiscale pyramids for efficient access
    - Timestamps use format YYYYMMDD_HHMMSS for versioning
    - UIDs are short hex strings for uniqueness
    - Paths in .ispy are relative to the project folder
    """).replace('"VERSION_PLACEHOLDER"', f'"{_version}"')


# ============================
# New InSituPy-Specific Tools
# ============================


@mcp.tool()
def get_datasets_guide() -> str:
    """Document sample datasets available in insitupy and how to load them.

    Returns:
        Available dataset functions with descriptions, usage examples,
        and guidance on when to use sample data.
    """
    lines: list[str] = ["# InSituPy Sample Datasets (insitupy.datasets)\n"]
    lines.append(
        "Sample datasets are provided for tutorials, testing, and exploration.\n"
        "They are downloaded on first call and cached locally.\n"
    )

    try:
        import insitupy.datasets.datasets as _ds

        lines.append("## Available Datasets\n")
        for name, obj in _safe_get_members(_ds, inspect.isfunction):
            if not _is_public(name) or obj.__module__ != _ds.__name__:
                continue
            sig = _format_signature(obj)
            desc = _short_doc(obj)
            lines.append(f"  {name}{sig}")
            if desc:
                lines.append(f"    {desc}")
        lines.append("")
    except Exception as exc:
        lines.append(f"(Could not introspect datasets module: {exc})\n")

    lines.append(textwrap.dedent("""\
    ## Usage

    ```python
    import insitupy as ispy

    # Download and load a Xenium breast cancer demo dataset
    data = ispy.datasets.xenium_human_breast_cancer()

    # Load a Visium dataset
    data = ispy.datasets.visium_human_breast_cancer()

    # Check what's already downloaded
    ispy.datasets.list_downloaded_datasets()

    # Specify a custom output directory
    data = ispy.datasets.xenium_human_breast_cancer(output_dir="/path/to/cache")

    # Force re-download if corrupted
    data = ispy.datasets.xenium_human_breast_cancer(overwrite=True)
    ```

    ## When to Use
    - Following tutorials or documentation examples
    - Writing and testing analysis workflows
    - Reproducing published results from InSituPy examples
    - Benchmarking performance on standardized data
    """))

    return _truncate("\n".join(lines), 3000)


@mcp.tool()
def get_result_types() -> str:
    """Document output objects returned by key analysis functions.

    Returns:
        Fields, methods, and usage of DiffExprResults and ImageRegistration.
    """
    lines: list[str] = ["# InSituPy Result Types\n"]

    # DiffExprResults
    lines.append("## DiffExprResults")
    lines.append("Returned by: ispy.tl.dge(), InSituExperiment.dge()\n")
    try:
        from insitupy.containers.results import DiffExprResults as _DER

        doc = inspect.getdoc(_DER)
        if doc:
            lines.append(_truncate(textwrap.indent(doc, "  "), 600))
            lines.append("")
        lines.append("Fields:")
        lines.append("  .main                  pd.DataFrame  — DGE results (target vs. reference)")
        lines.append(
            "  .target_neighborhood   Optional[pd.DataFrame]"
            "  — Target neighborhood DGE (if consider_neighbors=True)"
        )
        lines.append(
            "  .ref_neighborhood      Optional[pd.DataFrame]"
            "  — Reference neighborhood DGE (if consider_neighbors=True)"
        )
        lines.append(
            "  .config                DiffExprConfigCollector  — Analysis metadata and parameters"
        )
        lines.append("")
        lines.append("Methods:")
        for mname in ("get_all_results", "has_neighbors", "read", "save", "summary"):
            mobj = getattr(_DER, mname, None)
            if mobj is not None:
                sig = _format_signature(mobj)
                desc = _short_doc(mobj)
                lines.append(f"  .{mname}{sig}  — {desc}" if desc else f"  .{mname}{sig}")
        lines.append("")
    except Exception as exc:
        lines.append(f"  (Could not introspect DiffExprResults: {exc})\n")

    # ImageRegistration
    lines.append("## ImageRegistration")
    lines.append("Returned by: ispy.tl.register_images()\n")
    try:
        from insitupy.tools.registration import ImageRegistration as _IR

        doc = inspect.getdoc(_IR)
        if doc:
            lines.append(_truncate(textwrap.indent(doc, "  "), 500))
            lines.append("")
        lines.append("Key attributes (set after calling .run()):")
        lines.append(
            "  .T                   np.ndarray"
            "  — Estimated transformation matrix (2×3 affine or 3×3 homography)"
        )
        lines.append(
            "  .T_to_register       np.ndarray"
            "  — Transformation actually applied (may differ if image was resized)"
        )
        lines.append("  .registered          np.ndarray  — Warped (registered) image")
        lines.append("")
        lines.append("Key methods:")
        for mname in (
            "run",
            "calculate_transformation_matrix",
            "extract_features",
            "load_and_scale_images",
        ):
            mobj = getattr(_IR, mname, None)
            if mobj is not None:
                sig = _format_signature(mobj)
                desc = _short_doc(mobj)
                lines.append(f"  .{mname}{sig}  — {desc}" if desc else f"  .{mname}{sig}")
        lines.append("")
    except Exception as exc:
        lines.append(f"  (Could not introspect ImageRegistration: {exc})\n")

    lines.append(textwrap.dedent("""\
    ## Usage Example

    ```python
    # DGE result
    result = ispy.tl.dge(data, target_annotation_tuple=("tumor", "region1"))
    df = result.main           # pd.DataFrame with DGE columns
    result.summary()           # print a quick summary

    # ImageRegistration
    reg = ispy.tl.register_images(
        source=data_he, target=data_xenium,
        source_image="H&E", target_image="DAPI"
    )
    T = reg.T                  # transformation matrix
    # Apply to another image:
    data_he.transform(T, ...)
    ```
    """))

    return _truncate("\n".join(lines), 4000)


@mcp.tool()
def get_interactive_guide() -> str:
    """Guide for interactive napari-based visualization in InSituPy.

    Returns:
        Overview of data.show(), TranscriptViewerWidget, sync_geometries(),
        napari installation requirements, and typical interactive workflow.
    """
    lines: list[str] = ["# InSituPy Interactive Visualization\n"]
    lines.append('Requires napari: `pip install "napari[all]"`\n')

    # data.show()
    lines.append("## data.show() — Launch napari viewer")
    try:
        show_obj = _resolve_object("insitupy._core.data.InSituData.show")
        sig = _format_signature(show_obj)
        doc = inspect.getdoc(show_obj)
        lines.append(f"  InSituData.show{sig}\n")
        if doc:
            lines.append(textwrap.indent(_truncate(doc, 600), "  "))
        lines.append("")
    except Exception as exc:
        lines.append(f"  (Could not introspect InSituData.show: {exc})\n")

    # TranscriptViewerWidget
    lines.append("## TranscriptViewerWidget — Napari dock widget for transcripts")
    try:
        from insitupy.interactive._transcript_viewer import TranscriptViewerWidget

        sig = _format_signature(TranscriptViewerWidget.__init__)
        doc = inspect.getdoc(TranscriptViewerWidget)
        lines.append(f"  TranscriptViewerWidget.__init__{sig}\n")
        if doc:
            lines.append(_truncate(textwrap.indent(doc, "  "), 500))
        lines.append("")
    except Exception as exc:
        lines.append(f"  (Could not introspect TranscriptViewerWidget: {exc})\n")

    # sync_geometries
    lines.append("## sync_geometries() — Push napari edits back to InSituData")
    try:
        from insitupy.interactive.viewer import sync_geometries

        doc = inspect.getdoc(sync_geometries)
        lines.append("  ispy.interactive.viewer.sync_geometries()")
        if doc:
            lines.append(textwrap.indent(doc, "    "))
        else:
            lines.append(
                "    Reads Shapes/Points layers from the current napari viewer and"
            )
            lines.append(
                "    stores them as annotations or regions in the active InSituData."
            )
        lines.append("")
    except Exception as exc:
        lines.append(f"  (Could not introspect sync_geometries: {exc})\n")

    lines.append(textwrap.dedent("""\
    ## Installation

    ```bash
    # Full napari installation (recommended)
    pip install "napari[all]"

    # Or with PyQt5 backend explicitly
    pip install "napari[pyqt5]"
    ```

    ## Typical Interactive Workflow

    ```python
    import insitupy as ispy

    # 1. Load project
    data = ispy.InSituData.read("path/to/project/")
    data.load_images()
    data.load_cells()

    # 2. Launch napari viewer
    data.show(
        keys=["leiden", "EPCAM"],    # cell obs keys or gene names
        cells_layer="main",
        show_transcripts=True,        # transcript viewer widget
        transcript_lazy_loading=True  # recommended for >50M transcripts
    )

    # 3. Draw annotations in napari (Shapes layer named "@ ClassName (key)")
    # 4. Sync annotations back to the InSituData object
    ispy.interactive.viewer.sync_geometries()

    # 5. Assign annotations to cells
    data.assign_annotations(keys="all")
    ```

    ## Notes
    - `data.show()` raises ImportError if napari is not installed.
    - `transcript_lazy_loading=True` reduces memory (~1.5s delay per gene toggle).
    - `return_viewer=True` returns the napari viewer instance for programmatic control.
    """))

    return _truncate("\n".join(lines), 4000)


@mcp.tool()
def get_images_api() -> str:
    """List image I/O and utility functions in insitupy.images (insitupy.im).

    Returns:
        Functions from insitupy.images.io and insitupy.images.utils
        with signatures and descriptions; includes a note on dask-backed
        lazy loading behavior.
    """
    lines: list[str] = ["# InSituPy Images API (insitupy.im)\n"]

    lines.append(textwrap.dedent("""\
    ## Lazy Loading
    Images in InSituPy are stored as OME-Zarr pyramids and loaded as dask arrays.
    - Images are not read into RAM until `.compute()` is called or a region is accessed.
    - Pyramid levels allow efficient access at multiple resolutions.
    - `ImageData` stores multiple named images (e.g. "DAPI", "morphology_focus").
    - Use `is_from_disk()` / `is_from_zarr_disk()` to check if an array is still lazy.

    """))

    try:
        import insitupy.images.io as _im_io

        lines.append("## I/O (insitupy.images.io)")
        for name, obj in _safe_get_members(_im_io, inspect.isfunction):
            if not _is_public(name) or obj.__module__ != _im_io.__name__:
                continue
            sig = _format_signature(obj)
            desc = _short_doc(obj)
            ret = _returns_doc(obj)
            lines.append(f"  {name}{sig}")
            if desc:
                lines.append(f"    {desc}")
            if ret:
                lines.append(f"    Returns: {ret}")
        lines.append("")
    except Exception as exc:
        lines.append(f"(Could not introspect images.io: {exc})\n")

    try:
        import insitupy.images.utils as _im_utils

        lines.append("## Utilities (insitupy.images.utils)")
        for name, obj in _safe_get_members(_im_utils, inspect.isfunction):
            if not _is_public(name) or obj.__module__ != _im_utils.__name__:
                continue
            sig = _format_signature(obj)
            desc = _short_doc(obj)
            ret = _returns_doc(obj)
            lines.append(f"  {name}{sig}")
            if desc:
                lines.append(f"    {desc}")
            if ret:
                lines.append(f"    Returns: {ret}")
        lines.append("")
    except Exception as exc:
        lines.append(f"(Could not introspect images.utils: {exc})\n")

    return _truncate("\n".join(lines), 3000)


@mcp.tool()
def get_spatialdata_api() -> str:
    """Document SpatialData integration: convert_to_spatialdata and convert_from_spatialdata.

    Returns:
        Function signatures, parameter details, and usage patterns for
        converting between InSituPy and SpatialData formats.
    """
    lines: list[str] = ["# InSituPy SpatialData Integration (insitupy.spatialdata)\n"]
    lines.append("Requires the `spatialdata` package: `pip install spatialdata`\n")

    try:
        from insitupy.spatialdata.convert import (
            convert_from_spatialdata,
            convert_to_spatialdata,
        )

        for fn_name, fn_obj in [
            ("convert_to_spatialdata", convert_to_spatialdata),
            ("convert_from_spatialdata", convert_from_spatialdata),
        ]:
            sig = _format_signature(fn_obj)
            doc = inspect.getdoc(fn_obj)
            lines.append(f"## {fn_name}{sig}")
            if doc:
                lines.append(textwrap.indent(_truncate(doc, 600), "  "))
            lines.append("")
    except Exception as exc:
        lines.append(f"(Could not introspect spatialdata module: {exc})\n")

    lines.append(textwrap.dedent("""\
    ## Usage

    ```python
    import insitupy as ispy

    # Export to SpatialData
    data = ispy.InSituData.read("path/to/project/")
    sdata = ispy.spatialdata.convert_to_spatialdata(data, n_pyramids=3)

    # Import from SpatialData
    data = ispy.spatialdata.convert_from_spatialdata(
        sdata,
        image_data=("image", 0.2125),   # (image_key, pixel_size_µm)
        table_key="table",
        slide_id="my_slide",
    )
    ```

    ## Notes
    - `convert_to_spatialdata` wraps InSituData images, cell tables, and shapes.
    - `convert_from_spatialdata` reconstructs an InSituData from a SpatialData object.
    - The `image_data` parameter accepts a tuple or list of tuples: `(key, pixel_size)`.
    - Use `check_and_fix_case_insensitive_conflicts()` if SpatialData has key conflicts.
    """))

    return _truncate("\n".join(lines), 3000)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
