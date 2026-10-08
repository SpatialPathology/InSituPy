"""Console-script CLI for installing the bundled insitupy skill into an AI agent's skills
directory.

Distinct from ``insitupy-mcp`` (the MCP server entry point, unchanged): this CLI depends only
on the standard library, so a plain ``pip install insitupy-spatial`` (no ``[mcp]`` extra) makes
``insitupy install-skill`` available. It never auto-detects the calling agent - the destination
is always an explicit default or flag (see ``build_parser``). See
``insitupy/_ai/skill/SKILL.md`` for the skill content and ``tools/generate_skill_reference.py``
for how ``reference/*.md`` is produced.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from importlib import metadata
from importlib.resources import as_file, files
from pathlib import Path

_VERSION_RE = re.compile(r'^version:\s*"([^"]*)"', re.MULTILINE)

# Named destinations for `--target`. "agents" (the emerging agent-agnostic convention) is the
# no-flag default per the plan; project-local targets are relative to the current working
# directory, resolved at use time (not import time) so tests can control it via monkeypatching.
_TARGET_SUBPATHS: dict[str, tuple[str, Path]] = {
    "agents": ("cwd", Path(".agents/skills/insitupy-api")),
    "claude": ("home", Path(".claude/skills/insitupy-api")),
    "codex": ("home", Path(".codex/skills/insitupy-api")),
    "cursor": ("cwd", Path(".cursor/skills/insitupy-api")),
}


def _package_version() -> str:
    try:
        return metadata.version("insitupy-spatial")
    except metadata.PackageNotFoundError:
        return "unknown"


def _skill_version(skill_md: Path) -> str | None:
    if not skill_md.exists():
        return None
    match = _VERSION_RE.search(skill_md.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def _bundled_skill_resource():
    """Return the (possibly zip-backed) resource for the bundled skill tree."""
    return files("insitupy").joinpath("_ai", "skill")


def _resolve_destination(target: str, path: str | None) -> Path:
    if path is not None:
        return Path(path).expanduser().resolve() / "insitupy-api"
    root, subpath = _TARGET_SUBPATHS[target]
    base = Path.home() if root == "home" else Path.cwd()
    return (base / subpath).resolve()


def cmd_install_skill(args: argparse.Namespace) -> int:
    with as_file(_bundled_skill_resource()) as bundled_path:
        bundled_path = Path(bundled_path)
        bundled_skill_md = bundled_path / "SKILL.md"
        if not bundled_skill_md.exists():
            print(
                "The bundled skill is missing from this install "
                f"(expected {bundled_skill_md}).\n"
                "A real insitupy-spatial release always includes it - install the published "
                "package with `pip install insitupy-spatial`, or from a source checkout run "
                "`python tools/generate_skill_reference.py` first.",
                file=sys.stderr,
            )
            return 1

        destination = _resolve_destination(args.target, args.path)
        bundled_version = _skill_version(bundled_skill_md) or _package_version()
        existing_version = _skill_version(destination / "SKILL.md")

        if existing_version is not None and not args.force:
            print(
                f"A skill is already installed at {destination} (version {existing_version}).\n"
                f"Bundled version: {bundled_version}. Re-run with --force to overwrite.",
                file=sys.stderr,
            )
            return 1

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(bundled_path, destination, dirs_exist_ok=True)

    action = "Upgraded" if existing_version is not None else "Installed"
    print(f"{action} insitupy-api v{bundled_version} to {destination}")
    print("Upgrade later with: pip install -U insitupy-spatial && insitupy install-skill --force")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="insitupy", description="InSituPy AI-assistant helper CLI.")
    parser.add_argument("--version", action="store_true", help="Print the insitupy-spatial version and exit.")

    subparsers = parser.add_subparsers(dest="command")

    install = subparsers.add_parser(
        "install-skill",
        help="Copy the bundled insitupy-api skill into an AI agent's skills directory.",
    )
    install.add_argument(
        "--target",
        choices=sorted(_TARGET_SUBPATHS),
        default="agents",
        help="Named destination (default: agents -> ./.agents/skills/insitupy-api/).",
    )
    install.add_argument(
        "--path",
        default=None,
        help="Explicit destination directory; the skill is copied into <PATH>/insitupy-api/.",
    )
    install.add_argument("--force", action="store_true", help="Overwrite an already-installed skill.")
    install.set_defaults(func=cmd_install_skill)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(_package_version())
        return 0

    if not getattr(args, "command", None):
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
