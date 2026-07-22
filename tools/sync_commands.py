"""Render the canonical Claude review command + InSituPy pitfalls skill into Cursor/Codex.

Single source of truth lives under .claude/:
- .claude/commands/review.md  - the /review workflow
- .claude/skills/insitupy/     - the pitfalls skill (SKILL.md + references/)

Only the mandated /review workflow and the pitfalls knowledge are shared cross-agent; /plan
and /implement stay Claude-only. Each agent gets its correct current mechanism:
- Cursor: /review as a native command (.cursor/commands/review.md); pitfalls as a skill
  (.cursor/skills/insitupy/).
- Codex: no repo-scoped commands (custom prompts are deprecated), so /review is a skill
  (.codex/skills/insitupy-review/); pitfalls as a skill (.codex/skills/insitupy/).

Run `python tools/sync_commands.py` after editing either canonical source and commit the
regenerated output. `--check` verifies the mirrors are current and flags orphans (for a future
pre-commit/CI hook); `--dry-run` previews without writing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Canonical sources (authored once under .claude/).
REVIEW_COMMAND = REPO_ROOT / ".claude" / "commands" / "review.md"
PITFALLS_SKILL_DIR = REPO_ROOT / ".claude" / "skills" / "insitupy"

# Target roots. These are script-managed: the orphan scan walks them and removes any
# banner-carrying file this script no longer produces. Codex reads project skills from
# .codex/skills/ (repo root, scanned at session startup); Cursor reads .cursor/commands/
# (>= 1.6) and .cursor/skills/ (>= 2.4).
CURSOR_COMMANDS_DIR = REPO_ROOT / ".cursor" / "commands"
CURSOR_SKILLS_DIR = REPO_ROOT / ".cursor" / "skills"
CODEX_SKILLS_DIR = REPO_ROOT / ".codex" / "skills"
MANAGED_DIRS = (CURSOR_COMMANDS_DIR, CURSOR_SKILLS_DIR, CODEX_SKILLS_DIR)

BANNER_MARKER = "AUTO-GENERATED from"

# The pitfalls skill is a small fixed file set - copy these explicitly rather than walking
# an arbitrary tree.
PITFALLS_FILES = ("SKILL.md", "references/conventions_and_pitfalls.md")


def banner(source_rel: str) -> str:
    return (
        f"<!-- {BANNER_MARKER} {source_rel} by tools/sync_commands.py "
        "- do not edit here; edit the canonical file and re-run the sync. -->"
    )


def split_raw_frontmatter(text: str, source_rel: str) -> tuple[str, str]:
    """Split a leading ---delimited frontmatter block from the body, verbatim.

    Returns (frontmatter_block_including_both_delimiters, rest). The block is returned raw
    (not reparsed) so a copy preserves the original formatting. Raises ValueError on an
    opened-but-unterminated block rather than letting a stray `---` leak into the body.
    """
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return "", text
    i = 1
    while i < len(lines) and lines[i].strip() != "---":
        i += 1
    if i >= len(lines):
        raise ValueError(f"{source_rel}: frontmatter opened with '---' but never closed")
    frontmatter_block = "".join(lines[: i + 1])
    rest = "".join(lines[i + 1 :])
    return frontmatter_block, rest


def parse_frontmatter(frontmatter_block: str) -> dict[str, str]:
    """Parse flat `key: value` scalars from a raw frontmatter block (delimiters included)."""
    values: dict[str, str] = {}
    for line in frontmatter_block.splitlines():
        stripped = line.strip()
        if stripped == "---" or ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        values[key.strip()] = value.strip()
    return values


def yaml_double_quote(value: str) -> str:
    """Return value as a safely double-quoted YAML scalar (handles `:`, `#`, `[`, etc.)."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def ensure_trailing_newline(content: str) -> str:
    return content if content.endswith("\n") else content + "\n"


def render_cursor_command() -> dict[Path, str]:
    """Cursor command = body only (no frontmatter; Cursor commands are plain markdown)."""
    source_rel = ".claude/commands/review.md"
    _, body = split_raw_frontmatter(REVIEW_COMMAND.read_text(encoding="utf-8"), source_rel)
    content = f"{banner(source_rel)}\n\n{body.lstrip(chr(10)).rstrip()}\n"
    return {CURSOR_COMMANDS_DIR / "review.md": content}


def render_codex_review_skill() -> dict[Path, str]:
    """Codex has no repo-scoped commands, so /review ships as a skill (SKILL.md wrapper)."""
    source_rel = ".claude/commands/review.md"
    frontmatter_block, body = split_raw_frontmatter(
        REVIEW_COMMAND.read_text(encoding="utf-8"), source_rel
    )
    description = parse_frontmatter(frontmatter_block).get("description", "")
    skill_frontmatter = "\n".join(
        [
            "---",
            "name: insitupy-review",
            f"description: {yaml_double_quote(description)}",
            "---",
        ]
    )
    content = f"{skill_frontmatter}\n{banner(source_rel)}\n\n{body.lstrip(chr(10)).rstrip()}\n"
    return {CODEX_SKILLS_DIR / "insitupy-review" / "SKILL.md": content}


def copy_with_banner(source_file: Path, source_rel: str) -> str:
    """Copy a skill file verbatim with a banner injected after any frontmatter block."""
    text = source_file.read_text(encoding="utf-8")
    frontmatter_block, rest = split_raw_frontmatter(text, source_rel)
    if frontmatter_block:
        body = rest.lstrip("\n")
        return ensure_trailing_newline(f"{frontmatter_block}{banner(source_rel)}\n\n{body}")
    body = text.lstrip("\n")
    return ensure_trailing_newline(f"{banner(source_rel)}\n\n{body}")


def render_pitfalls_copies() -> dict[Path, str]:
    """Copy the canonical pitfalls skill into each agent's skills dir."""
    targets: dict[Path, str] = {}
    for skills_root in (CURSOR_SKILLS_DIR, CODEX_SKILLS_DIR):
        for rel in PITFALLS_FILES:
            source_file = PITFALLS_SKILL_DIR / Path(rel)
            source_rel = f".claude/skills/insitupy/{rel}"
            targets[skills_root / "insitupy" / Path(rel)] = copy_with_banner(
                source_file, source_rel
            )
    return targets


def build_targets() -> dict[Path, str]:
    targets: dict[Path, str] = {}
    targets.update(render_cursor_command())
    targets.update(render_codex_review_skill())
    targets.update(render_pitfalls_copies())
    return {path: ensure_trailing_newline(content) for path, content in targets.items()}


def find_stale(targets: dict[Path, str]) -> list[Path]:
    stale = []
    for path, content in targets.items():
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            stale.append(path)
    return stale


def find_orphans(targets: dict[Path, str]) -> list[Path]:
    """Banner-carrying files in a managed dir that this script no longer produces."""
    target_paths = set(targets)
    orphans = []
    for managed in MANAGED_DIRS:
        if not managed.exists():
            continue
        for path in sorted(managed.rglob("*")):
            if not path.is_file() or path in target_paths:
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if BANNER_MARKER in content:
                orphans.append(path)
    return orphans


def prune_empty_dirs() -> None:
    for managed in MANAGED_DIRS:
        if not managed.exists():
            continue
        for path in sorted(managed.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify generated files match the canonical sources and flag orphans; "
        "exit non-zero on any drift. No writes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change (including orphans to remove) without writing.",
    )
    args = parser.parse_args()

    targets = build_targets()
    stale = find_stale(targets)
    orphans = find_orphans(targets)

    if args.check:
        if stale or orphans:
            if stale:
                print("Stale or missing generated files:")
                for path in stale:
                    print(f"  {path.relative_to(REPO_ROOT)}")
            if orphans:
                print("Orphaned generated files (no longer produced):")
                for path in orphans:
                    print(f"  {path.relative_to(REPO_ROOT)}")
            return 1
        print("All generated files are up to date.")
        return 0

    if args.dry_run:
        if not stale and not orphans:
            print("No changes.")
            return 0
        if stale:
            print("Would write:")
            for path in stale:
                print(f"  {path.relative_to(REPO_ROOT)}")
        if orphans:
            print("Would remove (orphaned):")
            for path in orphans:
                print(f"  {path.relative_to(REPO_ROOT)}")
        return 0

    for path, content in targets.items():
        if path.exists() and path.read_text(encoding="utf-8") == content:
            continue  # write only on diff
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
    for path in orphans:
        path.unlink()
    prune_empty_dirs()
    print(
        f"Synced {len(targets)} generated files "
        f"({len(stale)} written, {len(orphans)} orphans removed)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
