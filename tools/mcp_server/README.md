# InSituPy MCP Server

An [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server that lets AI assistants explore and understand the InSituPy API and codebase on demand.

## Tools

### Generic Introspection

| Tool | Description |
|------|-------------|
| `list_modules` | List submodules/subpackages with descriptions |
| `list_classes` | List classes in a module with base classes |
| `list_functions` | List public functions with signatures |
| `get_class_info` | Detailed class info: methods, properties, docstring |
| `get_function_source` | Source code of any function/method/class |
| `get_docstring` | Full docstring for any module/class/function |
| `search_codebase` | Regex search across all source and test files |
| `list_test_files` | List all test files with descriptions |
| `read_source_file` | Read a source file with line numbers |

### InSituPy-Specific

| Tool | Description |
|------|-------------|
| `get_data_model` | Data model hierarchy and relationships |
| `get_io_formats` | Supported I/O formats and reader functions |
| `get_plotting_api` | Plotting functions grouped by category |
| `get_preprocessing_api` | Preprocessing functions by target type |
| `get_tools_api` | Analysis tools (DGE, distance, registration, ...) |
| `get_public_api` | Top-level namespace exports and submodule shorthands |
| `get_workflow_guide` | Step-by-step workflow examples |
| `get_storage_format` | On-disk storage architecture and metadata schema |
| `get_datasets_guide` | Available sample datasets and how to download them |
| `get_result_types` | Result objects returned by analysis tools |
| `get_interactive_guide` | napari-based interactive visualization |
| `get_images_api` | Image I/O and utility functions |
| `get_spatialdata_api` | SpatialData conversion functions |

## Setup

### Recommended: zero-install via `uvx`

No repository clone or manual environment setup needed. Add the following to your MCP client config:

```json
{
  "mcpServers": {
    "insitupy": {
      "command": "uvx",
      "args": ["--python", "3.12", "--from", "insitupy-spatial[mcp]", "insitupy-mcp"]
    }
  }
}
```

`uvx` (part of [uv](https://docs.astral.sh/uv/)) downloads and runs the server in an isolated environment automatically. Install `uv` first if you haven't already — see [uv installation](https://docs.astral.sh/uv/getting-started/installation/).

For client-specific config file paths and step-by-step instructions, see **[MCP_TUTORIAL.md](../../MCP_TUTORIAL.md)**.

### Local development setup

Use this approach if you want to run the server from a local repository clone — for example, to test unreleased changes.

**1. Clone the repository:**

```bash
git clone https://github.com/SpatialPathology/InSituPy.git
cd InSituPy
```

**2. Install with MCP dependencies:**

```bash
uv venv .venv --python 3.12
uv pip install -e ".[mcp]"
```

**3. Verify:**

```bash
# Windows
.venv\Scripts\insitupy-mcp.exe

# macOS/Linux
.venv/bin/insitupy-mcp
```

The server should start. Press Ctrl+C to stop.

**4. Configure your MCP client** using the local executable path:

```json
{
  "mcpServers": {
    "insitupy": {
      "command": "/path/to/InSituPy/.venv/bin/insitupy-mcp",
      "args": []
    }
  }
}
```

On Windows, replace with `.venv\Scripts\insitupy-mcp.exe`.

## How It Works

The server uses Python's `inspect`, `importlib`, and `ast` modules to introspect the installed `insitupy` package at runtime. Source file paths are derived from `insitupy.__file__`, so no hardcoded paths are needed.

All tool outputs are truncated to 2000 characters (4000 for source files) to keep responses concise.

**Note:** When installed in editable mode (`-e`), any changes to the InSituPy source are immediately reflected without reinstalling.
