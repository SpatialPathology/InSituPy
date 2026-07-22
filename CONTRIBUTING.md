# Contributing to InSituPy

Thanks for your interest in improving InSituPy! Contributions of all kinds are welcome: bug
reports, feature ideas, documentation, and code.

InSituPy is a Python framework for histology-guided, multi-sample analysis of single-cell spatial
transcriptomics data. It is under active development and maintained by a small team, so a little
structure goes a long way.

## Ways to contribute

- **Report a bug or request a feature:** open an
  [issue](https://github.com/SpatialPathology/InSituPy/issues).
- **Ask a question / discuss:** join our [Zulip chat](https://insitupy.zulipchat.com).
- **Contribute code or docs:** open a pull request (see below).

## Development setup

InSituPy requires Python >= 3.12 (>= 3.13 if you use the SpatialData integration).

```bash
# 1. Fork and clone
git clone https://github.com/<your-user>/InSituPy.git
cd InSituPy

# 2. Create an environment (conda shown; venv/uv are fine too)
conda create --name insitupy-dev python=3.13
conda activate insitupy-dev

# 3. Editable install with development dependencies
pip install -e ".[dev]"
```

Optional extras: `.[mcp]` (the AI MCP server), `.[spatialdata]` (SpatialData support). Combine as
needed, e.g. `pip install -e ".[dev,mcp]"`.

## Running tests and linting

```bash
# Run the test suite (or a targeted file while iterating)
pytest
pytest tests/test_<area>.py

# Lint and format the files you changed (ruff is configured in pyproject.toml)
ruff format path/to/changed_file.py
ruff check path/to/changed_file.py
```

Notes:
- **Lint the files you touched, and don't introduce new violations.** The codebase currently
  carries pre-existing ruff findings that are being cleaned up separately, so a repo-wide
  `ruff check` is not yet clean - don't be alarmed by unrelated findings, just keep your own
  changes clean.
- Prefer running the **targeted** test files for the area you changed while iterating; the full
  suite is slow.
- Some napari/Qt tests may fail in a headless environment for reasons unrelated to your change -
  mention it in your PR if you hit those.

## AI-assisted contributions

Using an AI assistant is encouraged, but it comes with responsibilities. **Please read
[AI_POLICY.md](AI_POLICY.md) before submitting AI-assisted work.** In short:

- You are fully responsible for everything you submit and must understand it.
- Before opening a PR, run an AI-assisted review over your changes and **paste a short summary of
  what it found into the PR**. The project provides a `/review` workflow (a self-contained InSituPy
  review command usable from a fresh clone across common AI coding agents) to standardize this.
- To help your assistant write correct InSituPy code, point it at the InSituPy **skill** (easiest)
  or the **MCP server** (power users). See the [README](README.md#ai-assistant-integration-mcp-server).

## Pull request checklist

1. Branch from the appropriate base branch and keep changes focused.
2. Tests pass (`pytest`), and `ruff check` / `ruff format` on the files you changed report no
   new issues.
3. Add or update tests for the behavior you changed.
4. If you used AI, follow [AI_POLICY.md](AI_POLICY.md) and include the AI-review summary.
5. Describe **what** changed and **why** in the PR description.

## License

By contributing, you agree that your contributions are licensed under the project's
[BSD-3-Clause](LICENSE) license.
