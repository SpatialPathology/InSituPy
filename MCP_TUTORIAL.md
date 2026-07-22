# InSituPy MCP Server Setup

## What this is

The InSituPy MCP server exposes the package's API (docstrings, function source, class info, workflow guides, etc.) as tools an MCP client (e.g. Claude Desktop) can query directly, so you don't need to manually browse the source or docs while coding. It also assists with analysis using InSituPy.

Two setups: `uvx` (published package, no local repo) and `.venv` (local dev, editable install).

## Option A: uvx (published package)

Use this if you don't need local code changes.

```json
"insitupy": {
  "command": "uvx",
  "args": ["--python", "3.13", "--from", "insitupy-spatial[mcp]", "insitupy-mcp"]
}
```

No install step needed, `uvx` fetches and runs it on demand.

## Option B: .venv (local dev)

Use this to test unreleased/dev branch changes.

### 1. Create venv

From the repo root (e.g. `Github\insitupy`):

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install package (editable, with MCP extras)

```powershell
pip install -e ".[mcp]"
```

`-e` makes local code changes take effect immediately, no reinstall needed.

### 3. Configure Claude Desktop

Edit `claude_desktop_config.json`:

```json
"insitupy": {
  "command": "C:\\Users\\<user>\\Github\\insitupy\\.venv\\Scripts\\insitupy-mcp.exe",
  "args": []
}
```

### 4. Link to a development branch

Check out the branch before or after installing (editable install tracks whatever is checked out):

```powershell
git checkout release/0.12.x
```

No reinstall required after switching branches, the editable install picks up the current checkout automatically.

## Restart Claude Desktop

Required for config/venv changes to take effect.

## Codex CLI

Codex reads MCP config from `~/.codex/config.toml` and supports stdio servers - the same two setups as above, only the config file and format differ (TOML instead of the JSON shown for Claude Desktop). This Codex configuration follows Codex's documented MCP setup but has not yet been tested end-to-end with InSituPy - feedback is welcome.

**uvx (published package).** Add the server from the terminal:

```bash
codex mcp add insitupy -- uvx --python 3.13 --from insitupy-spatial[mcp] insitupy-mcp
```

or write the equivalent block to `~/.codex/config.toml` directly:

```toml
[mcp_servers.insitupy]
command = "uvx"
args = ["--python", "3.13", "--from", "insitupy-spatial[mcp]", "insitupy-mcp"]
```

**.venv (local dev).** Create the virtual environment as in **Option B** above, then point Codex at its executable:

```toml
[mcp_servers.insitupy]
command = "C:\\Users\\<user>\\Github\\insitupy\\.venv\\Scripts\\insitupy-mcp.exe"
args = []
```

Start a new Codex session for changes to take effect. The full-install requirement and stable-path note below apply here too.

## Notes

- Full package install is required, not just MCP deps. Server tools (`get_docstring`, `get_function_source`, `list_classes`, etc.) introspect the actual `insitupy` package at runtime.
- Keep repo path stable. Renaming the repo folder later means updating `command` in the config.
