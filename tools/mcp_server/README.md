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
| `search_codebase` | Regex search across all source files |
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

## Setup

### 1. Install InSituPy with the MCP extra

```bash
pip install -e ".[mcp]"
```

Or install `mcp` separately:

```bash
pip install "mcp[cli]>=1.0.0"
```

### 2. Configure your MCP client

#### Claude Code (auto-discovery)

The `.mcp.json` file in the repository root enables auto-discovery. Just open the repo in Claude Code and the server will be available.

#### Claude Desktop

Add this to your Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "insitupy": {
      "command": "python",
      "args": ["-m", "tools.mcp_server"],
      "cwd": "/path/to/InSituPy"
    }
  }
}
```

#### Other MCP clients

Run the server via stdio:

```bash
cd /path/to/InSituPy
python -m tools.mcp_server
```

### 3. Verify

Test that the server starts correctly:

```bash
python -c "from tools.mcp_server.server import mcp; print('OK')"
```

## How It Works

The server uses Python's `inspect`, `importlib`, and `ast` modules to introspect the installed `insitupy` package at runtime. Source file paths are derived from `insitupy.__file__`, so no hardcoded paths are needed.

All tool outputs are truncated to 2000 characters (4000 for source files) to keep responses concise.
