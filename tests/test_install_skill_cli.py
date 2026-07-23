from pathlib import Path

import pytest

from insitupy._ai import cli


def test_install_skill_writes_to_explicit_path_and_reports_it(tmp_path, capsys):
    """Integration altitude: resolves the real in-package insitupy/_ai/skill/ tree."""
    bundled = Path(cli._bundled_skill_resource())
    if not (bundled / "SKILL.md").exists():
        pytest.skip(f"bundled skill not present at {bundled} in this install")

    exit_code = cli.main(["install-skill", "--path", str(tmp_path)])
    out = capsys.readouterr().out

    assert exit_code == 0
    destination = tmp_path / "insitupy"
    assert (destination / "SKILL.md").exists()
    assert (destination / "reference").is_dir()
    assert str(destination) in out


def test_install_skill_refuses_without_force_then_overwrites_with_force(tmp_path, capsys):
    bundled = Path(cli._bundled_skill_resource())
    if not (bundled / "SKILL.md").exists():
        pytest.skip(f"bundled skill not present at {bundled} in this install")

    assert cli.main(["install-skill", "--path", str(tmp_path)]) == 0
    installed_md = tmp_path / "insitupy" / "SKILL.md"
    original_content = installed_md.read_text(encoding="utf-8")

    # Tamper with the installed copy so an overwrite is observable.
    installed_md.write_text(original_content + "\nTAMPERED\n", encoding="utf-8")

    exit_code = cli.main(["install-skill", "--path", str(tmp_path)])
    assert exit_code != 0
    assert "TAMPERED" in installed_md.read_text(encoding="utf-8")

    exit_code = cli.main(["install-skill", "--path", str(tmp_path), "--force"])
    assert exit_code == 0
    assert "TAMPERED" not in installed_md.read_text(encoding="utf-8")


def test_install_skill_fails_clearly_when_bundled_skill_missing(tmp_path, monkeypatch, capsys):
    empty_source = tmp_path / "empty_bundled_skill"
    monkeypatch.setattr(cli, "_bundled_skill_resource", lambda: empty_source)

    exit_code = cli.main(["install-skill", "--path", str(tmp_path / "dest")])

    assert exit_code != 0
    assert "missing" in capsys.readouterr().err.lower()
