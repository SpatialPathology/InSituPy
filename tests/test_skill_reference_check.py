"""Tests for the ``--check-calls`` call-binding gate in tools/generate_skill_reference.py.

The gate is the regression guard that keeps the shipped AI skill / llms.txt / MCP curated text
from documenting calls that do not exist (the U-B1 class of drift, e.g. a fabricated
``register_images(source=..., target=...)``). These tests pin its real failure mode:

- the corrected curated renderers all bind against the live API (exit 0), and
- a fabricated keyword argument on a real insitupy function is caught,

while an unresolvable callee (a scanpy call, an unknown module) is skipped, never failed.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

# The generator imports tools/mcp_server/server.py, which needs the [mcp] extra.
gsr = pytest.importorskip(
    "generate_skill_reference",
    reason="requires the [mcp] extra (mcp_server.server import)",
)


def _messages(snippet: str) -> list[str]:
    """Run the gate's per-call binder over a standalone code snippet and return every
    failure message it produces."""
    tree = ast.parse(snippet)
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            msg = gsr._check_call_node(node)
            if msg:
                out.append(msg)
    return out


def test_curated_renderers_bind():
    """Every resolvable call in the live curated renderers / SKILL.md / llms.txt binds."""
    assert gsr.check_calls(gsr.resolve_version()) == 0


def test_fabricated_keyword_is_caught():
    """A fabricated keyword on a real insitupy function fails to bind."""
    # register_images has no `source`/`target` params (the original U-B1 drift).
    msgs = _messages('ispy.tl.register_images(source=he, target=xen)')
    assert msgs, "expected a bind failure for fabricated register_images kwargs"
    assert "register_images" in msgs[0]


def test_fabricated_constructor_kwarg_is_caught():
    """CellData(matrix=...) is drift: the parameter is `table`."""
    msgs = _messages('CellData(matrix=x)')
    assert msgs, "expected a bind failure for CellData(matrix=...)"


def test_valid_calls_pass():
    """Correct module-level, constructor, and instance-method calls bind cleanly."""
    good = "\n".join(
        [
            'ispy.tl.dge(data, target_annotation_tuple=("tumor", "region1"))',
            'ispy.pl.spatial(data, keys="leiden", image_key="DAPI")',
            'data.cells.add_baysor("xenium/", "baysor/", pixel_size=0.2125)',
            'data.crop(xlim=(0, 100), ylim=(0, 100))',
        ]
    )
    assert _messages(good) == []


def test_unresolvable_callees_are_skipped():
    """Non-insitupy calls and unknown names are skipped, never failed (no false positives)."""
    skipped = "\n".join(
        [
            'sc.pp.normalize_total(adata)',          # scanpy: unknown root
            'some_unknown_module.func(bogus=1)',      # cannot resolve
            'plt.show()',                             # matplotlib: unknown root
        ]
    )
    assert _messages(skipped) == []
