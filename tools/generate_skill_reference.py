"""Generate the user-facing insitupy skill's static reference layer + llms.txt.

Single source of truth: the curated renderer functions in ``tools/mcp_server/server.py`` (the
`insitupy` MCP server). This script imports and calls those renderers directly - it does not
duplicate their content - so the MCP server and the skill's static ``reference/`` files are two
render targets of one generator, not two hand-maintained copies. The ``@mcp.tool()`` decorator
returns the wrapped function unchanged, so calling e.g. ``server.get_public_api()`` here is
identical to what the live MCP tool returns.

Requires the ``[mcp]`` extra (and a full insitupy environment, since ``mcp_server.server``
imports insitupy) - acceptable because this script runs only in CI-on-tag and the maintainer's
version-bump workflow, never on a user's machine.

The generated pitfalls copy (``reference/conventions_and_pitfalls.md``) is sourced from
``.claude/skills/insitupy-pitfalls/reference/conventions_and_pitfalls.md`` - the same canonical file
that ``tools/sync_commands.py`` copies into ``.cursor/`` and ``.codex/``. Edit the canonical
file, not this generator's output.

The stamped version comes from ``pyproject.toml``, the declared single source of truth - not from
``insitupy.__version__``. See ``resolve_version`` for why that distinction is load-bearing.

Usage:
    python tools/generate_skill_reference.py                     # regenerate + stamp version
    python tools/generate_skill_reference.py --check              # verify no drift; exit 1 on drift
    python tools/generate_skill_reference.py --check-symbols       # verify curated symbols resolve
    python tools/generate_skill_reference.py --check-calls         # bind every curated call; exit 1 on a bad call

``--check-calls`` extracts every fenced ``python`` block from the curated renderer outputs
(plus ``SKILL.md`` and ``llms.txt``), parses it with ``ast``, and for each function/method call
it can statically resolve against the live ``insitupy`` package, checks the documented arguments
against the real signature with ``inspect.signature(...).bind_partial(...)``. A ``TypeError``
(unknown keyword, too many positionals) is a hard failure. It is a *best-effort* binder: a callee
it cannot resolve (a scanpy call, an instance-method receiver whose type it does not know) is
skipped, never failed - so it never false-positives, it only fails on a *resolvable* callee with
an *impossible* binding. This is the regression guard for the U-B1 class of drift (e.g. a
fabricated ``register_images(source=..., target=...)`` call). Fabricated *names* under a real
module are out of scope here (they don't resolve, so they're skipped) - ``--check-symbols`` and
the drift sweep cover those.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import re
import sys
import textwrap
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
SKILL_DIR = REPO_ROOT / "insitupy" / "_ai" / "skill"
SKILL_MD = SKILL_DIR / "SKILL.md"
REFERENCE_DIR = SKILL_DIR / "reference"
LLMS_TXT = REPO_ROOT / "llms.txt"
PITFALLS_SOURCE = REPO_ROOT / ".claude" / "skills" / "insitupy-pitfalls" / "reference" / "conventions_and_pitfalls.md"
PITFALLS_SOURCE_REL = ".claude/skills/insitupy-pitfalls/reference/conventions_and_pitfalls.md"

# So `import mcp_server` works from a bare source checkout too, not only an installed env.
sys.path.insert(0, str(REPO_ROOT / "tools"))

from mcp_server import server as S  # noqa: E402

import insitupy  # noqa: E402

# reference filename -> (renderer function name, one-line "use when" blurb)
RENDERERS: dict[str, tuple[str, str]] = {
    "public_api.md": ("get_public_api", "top-level namespace + submodule shorthands"),
    "data_model.md": ("get_data_model", "container hierarchy and relationships"),
    "workflows.md": ("get_workflow_guide", "full step-by-step recipes"),
    "io_formats.md": ("get_io_formats", "readers, native format, import methods"),
    "storage_format.md": ("get_storage_format", "on-disk .ispy layout"),
    "preprocessing.md": ("get_preprocessing_api", "pp functions + signatures"),
    "plotting.md": ("get_plotting_api", "pl functions by category"),
    "tools.md": ("get_tools_api", "tl DGE/distance/neighbors/registration"),
    "result_types.md": ("get_result_types", "DiffExprResults / ImageRegistration fields"),
    "spatialdata.md": ("get_spatialdata_api", "convert to/from SpatialData"),
    "images.md": ("get_images_api", "im I/O + lazy-loading note"),
    "datasets.md": ("get_datasets_guide", "sample datasets"),
    "interactive.md": ("get_interactive_guide", "napari .show() workflow"),
}

# Curated load-bearing symbols named in SKILL.md / llms.txt / the MCP renderers' hardcoded
# lists (e.g. get_plotting_api's categories dict). --check-symbols asserts each still resolves
# against the live installed package, catching the case where a hand-curated reference names a
# symbol the API no longer exports. Verified against the live API at implementation time.
CURATED_SYMBOLS: tuple[str, ...] = (
    "insitupy.io.read_xenium",
    "insitupy.io.read_visium",
    "insitupy.io.read_qupath",
    "insitupy._core.data.InSituData.read",
    "insitupy._core.data.InSituData.saveas",
    "insitupy._core.data.InSituData.save",
    "insitupy._core.data.InSituData.crop",
    "insitupy._core.data.InSituData.load_cells",
    "insitupy._core.data.InSituData.load_images",
    "insitupy._core.data.InSituData.load_transcripts",
    "insitupy._core.data.InSituData.show",
    "insitupy._core.data.InSituData.import_annotations",
    "insitupy._core.data.InSituData.assign_annotations",
    "insitupy.pp.normalize_and_transform",
    "insitupy.pp.filter_cells",
    "insitupy.pp.cluster_cells",
    "insitupy.tl.dge",
    "insitupy.experiment.data.InSituExperiment.dge",
    "insitupy.tl.register_images",
    "insitupy.pl.spatial",
    "insitupy.pl.umap",
    "insitupy.pl.volcano",
    "insitupy.spatialdata.convert_to_spatialdata",
    "insitupy.spatialdata.convert_from_spatialdata",
    "insitupy.experiment.data.InSituExperiment.add",
    "insitupy.experiment.data.InSituExperiment.to_anndata",
    "insitupy.datasets.xenium_human_breast_cancer",
)

_BANNER_VERSION_RE = re.compile(r"\(insitupy v([^)]+)\)")
_SKILL_VERSION_LINE_RE = re.compile(r'^version:\s*".*?"(?:\s*#.*)?$', re.MULTILINE)


def _banner(source_desc: str, version: str) -> str:
    return (
        f"<!-- AUTO-GENERATED by tools/generate_skill_reference.py from {source_desc} "
        f"(insitupy v{version}) - do not edit here; edit the source and re-run the "
        "generator. -->"
    )


def _drop_leading_heading(text: str) -> str:
    """Drop a renderer's own leading '# Title' line and demote its '##' headings by one
    level, so embedding it under a caller-supplied '##' heading nests correctly instead of
    colliding with it.
    """
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    demoted = [f"#{line}" if line.startswith("#") else line for line in lines]
    return "\n".join(demoted).strip()


def render_reference_file(filename: str, renderer_name: str, version: str) -> str:
    renderer = getattr(S, renderer_name)
    body = renderer().rstrip()
    banner = _banner(f"mcp_server.server.{renderer_name}()", version)
    return f"{banner}\n\n{body}\n"


def render_pitfalls_copy(version: str) -> str:
    text = PITFALLS_SOURCE.read_text(encoding="utf-8")
    banner = _banner(PITFALLS_SOURCE_REL, version)
    body = text.lstrip("\n").rstrip()
    return f"{banner}\n\n{body}\n"


_LLMS_TXT_TEMPLATE = textwrap.dedent("""\
    {banner}

    # InSituPy

    > InSituPy (insitupy-spatial on PyPI) is a Python framework for histology-guided,
    > multi-sample analysis of single-cell spatial transcriptomics data (10x Xenium, Visium,
    > QuPath). This file documents insitupy-spatial v{version}.

    ## Fallback contract

    If you are an AI agent with access to the `insitupy` MCP server, prefer its live tools over
    this file - they resolve against the actually installed source and never go stale. This
    file is the floor that works with no skill loader and no MCP server (e.g. plain web chat).

    ## Suspect a stale reference before concluding an API doesn't exist

    This file is versioned per release. If a function or class you need is missing here, check
    the installed version (`python -c "import insitupy; print(insitupy.__version__)"`) - if it
    is newer than v{version} above, re-fetch this file from the matching release instead of
    concluding the API doesn't exist.

    ## What InSituPy is

    A two-level data model: `InSituData` (a single sample - cells, images, transcripts,
    annotations, regions, units) and `InSituExperiment` (aggregates multiple `InSituData`
    instances with a sample-level metadata DataFrame for cross-sample analysis).

    ## The workflow arc

    ```python
    import insitupy as ispy

    data = ispy.io.read_xenium("path/to/xenium_output/")
    ispy.pp.normalize_and_transform(data)
    ispy.pp.cluster_cells(data)
    result = ispy.tl.dge(data, target_annotation_tuple=("tumor", "region1"))
    ispy.pl.spatial(data, keys="leiden", image_key="DAPI")
    data.saveas("path/to/my_project/")
    ```

    ## Public API index

    {public_api}

    ## Top pitfalls

    - `cells_layer` (which segmentation layer) vs `layer` (an `AnnData.layers` key) are
      different concepts sharing similar names - check both when a signature has both.
    - Never construct `InSituData(path=...)` directly on a saved project path - use
      `InSituData.read(path)` instead.
    - `pp.filter_cells` / `pp.filter_genes` accept exactly one criterion per call.
    - `inplace=False` is the house default across mutating methods - pass `inplace=True`
      explicitly for in-place mutation.
    - See the full skill's `reference/conventions_and_pitfalls.md` for the complete list.

    ## Links

    - Docs: https://insitupy.readthedocs.io
    - Repository: https://github.com/SpatialPathology/InSituPy
    - Skill ZIP (per release): https://github.com/SpatialPathology/InSituPy/releases/latest
    - MCP server setup: see `tools/mcp_server/README.md` in the repository
    """)


def render_llms_txt(version: str) -> str:
    public_api = _drop_leading_heading(S.get_public_api())
    banner = _banner("tools/generate_skill_reference.py", version)
    return _LLMS_TXT_TEMPLATE.format(banner=banner, version=version, public_api=public_api)


def stamp_skill_version(text: str, version: str) -> str:
    """Rewrite only the `version:` line in SKILL.md's frontmatter - single-line replace."""
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{SKILL_MD}: expected a leading '---' frontmatter block")
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            raise ValueError(f"{SKILL_MD}: frontmatter has no 'version:' field to stamp")
        if lines[i].startswith("version:"):
            lines[i] = (
                f'version: "{version}"   '
                "# AUTO-STAMPED by tools/generate_skill_reference.py to the package version\n"
            )
            return "".join(lines)
    raise ValueError(f"{SKILL_MD}: frontmatter never closed with '---'")


def _normalize_modulo_version(content: str, filename: str) -> str:
    """Strip version-only noise so --check compares content, not the version stamp.

    llms.txt embeds the version both in its banner and inline in the body (e.g. "v0.12.0b6"),
    so it isn't enough to normalize just the banner's parenthetical - blank every occurrence of
    the version string this particular file's banner was stamped with.
    """
    if filename == "SKILL.md":
        return _SKILL_VERSION_LINE_RE.sub('version: "X"', content)
    match = _BANNER_VERSION_RE.search(content)
    if not match:
        return content
    return content.replace(match.group(1), "X")


def build_targets(version: str) -> dict[Path, str]:
    targets: dict[Path, str] = {}
    for filename, (renderer_name, _use_when) in RENDERERS.items():
        targets[REFERENCE_DIR / filename] = render_reference_file(filename, renderer_name, version)
    targets[REFERENCE_DIR / "conventions_and_pitfalls.md"] = render_pitfalls_copy(version)
    targets[LLMS_TXT] = render_llms_txt(version)
    targets[SKILL_MD] = stamp_skill_version(SKILL_MD.read_text(encoding="utf-8"), version)
    return targets


def resolve_version() -> str:
    """Read the version to stamp from pyproject.toml, the declared single source of truth.

    Deliberately NOT ``insitupy.__version__``: that resolves through ``importlib.metadata``,
    which an editable install writes once at install time and never refreshes. Immediately after
    a version bump the installed metadata still reports the *previous* version, so stamping from
    it would silently label the skill with the wrong release - and ``--check`` normalizes the
    version stamp before comparing, so nothing downstream would catch it. Reading pyproject.toml
    removes the failure mode instead of relying on the maintainer remembering to reinstall.

    A mismatch is still worth reporting, because a stale install also means ``--check-symbols``
    would introspect old code.
    """
    with PYPROJECT.open("rb") as handle:
        version = tomllib.load(handle)["project"]["version"]

    installed = insitupy.__version__
    if installed != version:
        print(
            f"NOTE: pyproject.toml declares {version}, but the installed insitupy-spatial "
            f"reports {installed}. Stamping {version}. The installed package is stale - re-run "
            "`pip install -e . --no-deps` if you also need --check-symbols to see current code.",
            file=sys.stderr,
        )
    return version


def check_symbols() -> int:
    failures: list[str] = []
    for dotted_path in CURATED_SYMBOLS:
        try:
            S._resolve_object(dotted_path)
        except Exception as exc:
            failures.append(f"  {dotted_path}: {exc}")
    if failures:
        print(f"{len(failures)} curated symbol(s) no longer resolve against the live API:")
        print("\n".join(failures))
        return 1
    print(f"All {len(CURATED_SYMBOLS)} curated symbols resolve.")
    return 0


# --- --check-calls: static call-binding of curated example code ----------------------------

# Module-alias roots that stand for the insitupy package in curated snippets.
_MODULE_ALIASES: dict[str, str] = {"ispy": "insitupy", "isp": "insitupy", "insitupy": "insitupy"}

# Bare InSituPy class/callable names used directly in curated snippets (constructor or
# classmethod calls), mapped to a dotted path _resolve_object can resolve. Only unambiguous
# InSituPy names belong here.
_CLASS_ROOTS: dict[str, str] = {
    "InSituData": "insitupy._core.data.InSituData",
    "InSituExperiment": "insitupy.experiment.data.InSituExperiment",
    "CellData": "insitupy.containers.cell_data.CellData",
    "ImageData": "insitupy.containers.image_data.ImageData",
}

# Conventional instance-variable receivers in curated snippets, mapped to the class whose
# (unbound) methods we bind against. The implicit ``self`` is dropped before binding. Keep this
# small and conventional; a wrong entry could false-positive, which the "corrected renderers
# pass" test guards against.
_RECEIVER_CLASSES: dict[str, str] = {
    "data": "insitupy._core.data.InSituData",
    "xd": "insitupy._core.data.InSituData",
    "cells": "insitupy.containers.multi_cell_data.MultiCellData",
    "data.cells": "insitupy.containers.multi_cell_data.MultiCellData",
    "exp": "insitupy.experiment.data.InSituExperiment",
    "experiment": "insitupy.experiment.data.InSituExperiment",
}

_PY_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)

_BIND_DUMMY = object()  # opaque placeholder value for bind_partial


def _dotted_from_func(func: ast.AST) -> str | None:
    """Return the dotted call target (e.g. ``ispy.tl.dge``, ``data.cells.add_baysor``) for an
    ``ast.Call`` func node, or ``None`` if it is not a plain Name/Attribute chain (e.g. the call
    target is itself a call result or a subscript)."""
    parts: list[str] = []
    node = func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _resolve_callee(dotted: str) -> tuple[object, bool] | None:
    """Resolve *dotted* to a live callable, or ``None`` if it is not an insitupy callee this
    best-effort binder can resolve. Returns ``(obj, is_unbound_instance_method)``; the caller
    drops the implicit ``self`` for unbound instance methods. Any resolution error means "skip"."""
    parts = dotted.split(".")
    root = parts[0]

    if root in _MODULE_ALIASES:
        target = ".".join([_MODULE_ALIASES[root], *parts[1:]])
        return (S._resolve_object(target), False)

    if root in _CLASS_ROOTS:
        target = ".".join([_CLASS_ROOTS[root], *parts[1:]])
        return (S._resolve_object(target), False)

    receiver = ".".join(parts[:-1])
    if receiver in _RECEIVER_CLASSES:
        cls = S._resolve_object(_RECEIVER_CLASSES[receiver])
        return (getattr(cls, parts[-1]), True)

    return None


def _binding_signature(obj: object, is_unbound_instance_method: bool) -> inspect.Signature:
    """Signature to bind against. Drops the leading ``self``/``cls`` for an unbound instance
    method accessed off its class (bound methods and classmethods already exclude it; a class
    resolves to its ``__init__`` signature minus ``self``)."""
    sig = inspect.signature(obj)
    if is_unbound_instance_method and not inspect.ismethod(obj):
        params = list(sig.parameters.values())
        if params and params[0].name in ("self", "cls"):
            sig = sig.replace(parameters=params[1:])
    return sig


def _check_call_node(node: ast.Call) -> str | None:
    """Return a failure message if *node* is a resolvable insitupy call whose documented
    arguments cannot bind to the real signature, else ``None``."""
    dotted = _dotted_from_func(node.func)
    if dotted is None:
        return None
    # A starred positional (*args) or a ** spread means we cannot count arguments reliably.
    if any(isinstance(a, ast.Starred) for a in node.args):
        return None
    if any(kw.arg is None for kw in node.keywords):
        return None

    try:
        resolved = _resolve_callee(dotted)
    except Exception:
        return None  # unresolvable -> skip, never fail
    if resolved is None:
        return None
    obj, is_unbound_instance_method = resolved
    if not callable(obj):
        return None

    try:
        sig = _binding_signature(obj, is_unbound_instance_method)
    except (ValueError, TypeError):
        return None  # no introspectable signature -> skip

    n_positional = len(node.args)
    kw_names = [kw.arg for kw in node.keywords]
    try:
        sig.bind_partial(*([_BIND_DUMMY] * n_positional), **{name: _BIND_DUMMY for name in kw_names})
    except TypeError as exc:
        return f"{dotted}(...): {exc}"
    return None


def _iter_curated_code_blocks(version: str):
    """Yield the source of every fenced ``python`` block in the curated renderer outputs,
    ``llms.txt``, and ``SKILL.md``."""
    texts: list[str] = [getattr(S, renderer_name)() for renderer_name, _ in RENDERERS.values()]
    texts.append(render_llms_txt(version))
    texts.append(SKILL_MD.read_text(encoding="utf-8"))
    for text in texts:
        for match in _PY_BLOCK_RE.finditer(text):
            yield match.group(1)


def check_calls(version: str) -> int:
    failures: list[str] = []
    for block in _iter_curated_code_blocks(version):
        try:
            tree = ast.parse(block)
        except SyntaxError:
            continue  # not parseable as a standalone module -> skip the block
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                message = _check_call_node(node)
                if message:
                    failures.append(message)
    if failures:
        print(f"{len(failures)} curated call(s) do not bind against the live API:")
        print("\n".join(f"  {msg}" for msg in failures))
        return 1
    print("All resolvable curated calls bind against the live API.")
    return 0


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def check_drift(version: str) -> int:
    targets = build_targets(version)
    stale: list[Path] = []
    for path, content in targets.items():
        if not path.exists():
            stale.append(path)
            continue
        existing = path.read_text(encoding="utf-8")
        if _normalize_modulo_version(existing, path.name) != _normalize_modulo_version(
            content, path.name
        ):
            stale.append(path)
    if stale:
        print("Stale or missing generated files (content drift, ignoring version stamp):")
        for path in stale:
            print(f"  {_display_path(path)}")
        return 1
    print("No drift: generated reference is current.")
    return 0


def write_targets(version: str) -> None:
    targets = build_targets(version)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    for path, content in targets.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
    print(f"Wrote {len(targets)} files (insitupy v{version}).")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify generated files match the live renderers modulo the version stamp; "
        "exit non-zero on drift. No writes.",
    )
    parser.add_argument(
        "--check-symbols",
        action="store_true",
        help="Verify every curated load-bearing symbol resolves in the live package; "
        "exit non-zero if any is missing. No writes.",
    )
    parser.add_argument(
        "--check-calls",
        action="store_true",
        help="Statically bind every insitupy call in the curated example code against the live "
        "API; exit non-zero on a call that cannot bind (unknown keyword, too many positionals). "
        "No writes.",
    )
    args = parser.parse_args()

    if args.check_symbols:
        return check_symbols()

    version = resolve_version()

    if args.check_calls:
        return check_calls(version)

    if args.check:
        return check_drift(version)

    write_targets(version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
