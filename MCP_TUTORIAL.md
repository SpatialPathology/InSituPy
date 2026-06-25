# Using InSituPy with AI Assistants (MCP Server Tutorial)

This tutorial explains how to connect an AI assistant (Claude, Cursor, Windsurf, and others) to the InSituPy MCP server, so the AI can answer questions about InSituPy, explain its API, search the codebase, and help you write analysis code.

---

## What is the MCP Server?

**MCP** (Model Context Protocol) is an open standard that lets AI assistants call external tools. The InSituPy MCP server exposes 22 tools that give an AI assistant live access to:

- The full InSituPy API (functions, classes, signatures, docstrings)
- Searchable source code and test files
- Curated overviews of the data model, I/O formats, plotting, preprocessing, and analysis tools
- Ready-to-run workflow examples

Once connected, you can ask your AI assistant things like:

> "What arguments does `insitupy.io.read_xenium` accept?"

> "Show me a complete workflow for loading Xenium data and running clustering."

> "Search the codebase for how pseudobulk is implemented."

> "What file formats does InSituPy support for reading data?"

The AI will call the appropriate MCP tool and answer based on the actual, up-to-date InSituPy code rather than relying solely on its training data.

---

## Prerequisites

**`uv`** must be installed on your system. `uv` is a fast Python package manager that `uvx` (the zero-install runner) is part of.

Install `uv`:

- **macOS / Linux:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **macOS (Homebrew):**
  ```bash
  brew install uv
  ```
- **Windows and other options:** see [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

Once `uv` is installed, no further manual setup is needed — `uvx` downloads and runs the InSituPy MCP server automatically in an isolated environment.

> **Python version:** All config snippets below use `--python 3.12`. This is required because a transitive dependency of napari (`triangle`) does not yet ship pre-built wheels for Python 3.13+. Without this flag, uvx may pick a newer Python and fail to build `triangle` from source.

> **Advanced / local development:** If you prefer to run the server from a local repository clone (e.g. to test unreleased changes), see the [Local Development Setup](#local-development-setup) section at the end of this document.

---

## Setup by Client

### Claude Desktop

Claude Desktop has built-in MCP support. You configure servers via a JSON file.

**1. Find (or create) your Claude Desktop config file:**

| Platform | Path |
|----------|------|
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |
| Linux | `~/.config/Claude/claude_desktop_config.json` |

**2. Add the InSituPy server** to `mcpServers`:

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

This is all you need. `uvx` downloads `insitupy-spatial` with its MCP dependencies and launches the server in an isolated environment — no repository clone or manual Python setup required.

**3. Restart Claude Desktop.** The server starts automatically when Claude Desktop launches. You should see "insitupy" listed in the MCP tools panel (hammer icon).

---

### Claude Code (CLI)

There are two ways to use the InSituPy MCP server with Claude Code depending on whether you have the repository cloned locally.

**Option A — No repository clone (recommended for most users)**

Add the server to Claude Code's global MCP config (`~/.claude/mcp.json` on macOS/Linux, `%USERPROFILE%\.claude\mcp.json` on Windows):

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

The server will then be available in any Claude Code session, regardless of which directory you're working in.

**Option B — Working inside the InSituPy repository**

The repository already contains a `.mcp.json` file configured for Claude Code. When you run `claude` from inside the repository root, the server starts automatically. The `.mcp.json` uses `python -m tools.mcp_server`, which requires `python` to resolve to your InSituPy environment. Activate it first:

```bash
conda activate insitupy
cd /path/to/InSituPy
claude
```

To verify the server loaded, run `/mcp` in the Claude Code prompt and confirm "insitupy" appears in the list.

---

### Cursor

Cursor supports MCP servers via a JSON config file.

**1. Open Cursor Settings → MCP** (or edit the config file directly).

**Project-level config** — create `.cursor/mcp.json` in your project root:

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

**Global config** — edit `~/.cursor/mcp.json` (macOS/Linux) or `%USERPROFILE%\.cursor\mcp.json` (Windows) with the same content.

**2. Restart Cursor.** The MCP server will be available in Agent mode. Enable it in the chat panel if needed.

---

### Windsurf

Windsurf (by Codeium) reads MCP configs from:

| Platform | Path |
|----------|------|
| macOS / Linux | `~/.codeium/windsurf/mcp_config.json` |
| Windows | `%USERPROFILE%\.codeium\windsurf\mcp_config.json` |

Add the InSituPy server:

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

Restart Windsurf after saving. The server should appear in the Cascade agent panel.

---

### Continue.dev (VS Code / JetBrains)

Continue is an open-source AI coding assistant that supports MCP.

**1. Open your Continue config file** (`~/.continue/config.json` or via the Continue sidebar → gear icon).

**2. Add an MCP server** under the `"experimental"` key:

```json
{
  "experimental": {
    "modelContextProtocolServers": [
      {
        "transport": {
          "type": "stdio",
          "command": "uvx",
          "args": ["--python", "3.12", "--from", "insitupy-spatial[mcp]", "insitupy-mcp"]
        }
      }
    ]
  }
}
```

**3. Reload Continue** (reload VS Code window or restart JetBrains). The InSituPy tools will be available in the Chat panel when using an Agent model.

---

### Cline (VS Code Extension)

Cline is a VS Code extension with MCP support.

**1. Open Cline settings** → **MCP Servers** tab → **Add Server**.

**2. Choose "Command (stdio)"** and fill in:

- **Command:** `uvx`
- **Args:** `--python 3.12 --from insitupy-spatial[mcp] insitupy-mcp`
- **Name:** `insitupy`

Or edit `cline_mcp_settings.json` directly (accessible from the Cline MCP panel) with the same JSON structure as the other clients above.

---

### ChatGPT and Other Non-MCP Clients

> **Note:** The options below have **not been tested** by the InSituPy team, as we do not have access to a ChatGPT account with plugin/tool support. If you try either approach and run into issues — or if it works well — we'd love to hear from you. Please open an [issue](https://github.com/SpatialPathology/InSituPy/issues) or reach out via our [Zulip chat](https://insitupy.zulipchat.com).

ChatGPT and similar web-based assistants do not natively support MCP. There are two practical options:

**Option 1 — Use an MCP-to-HTTP bridge**

Tools such as [`mcp-proxy`](https://github.com/sparfenyuk/mcp-proxy) can expose a stdio MCP server as an HTTP/SSE endpoint. If your client supports OpenAPI-based tool use or custom plugins, you can connect it via such a bridge. Setup steps will vary depending on the bridge tool and client — consult the bridge tool's documentation.

**Option 2 — Copy relevant tool output manually**

Run the MCP server interactively using the MCP inspector:

```bash
# Install uv if not already installed, then run:
uvx --from "insitupy-spatial[mcp]" mcp dev insitupy-mcp
```

This opens a browser-based inspector where you can call individual tools and copy their output to use in any chat interface.

---

## Verifying the Connection

Once set up, test the connection by asking your AI assistant:

> "Use the InSituPy MCP tools to list all available submodules."

The assistant should call the `list_modules` tool and return a structured list of InSituPy submodules (`io`, `pl`, `pp`, `tl`, `im`, `interactive`, `datasets`, `spatialdata`, `utils`). If it responds without calling a tool or says it has no InSituPy tools available, check that:

1. `uv` is installed and `uvx` is available on your PATH (`uvx --version` should print a version number).
2. The `"command"` in your config is exactly `"uvx"` (not a full path).
3. You restarted the client after saving the config.

---

## Example Prompts

Once connected, try these prompts to get started:

| Goal | Example prompt |
|------|----------------|
| Understand the data model | "Describe the InSituPy data model and how `InSituData` and `InSituExperiment` relate to each other." |
| Read Xenium data | "Show me a complete workflow for loading a Xenium dataset and running standard single-cell analysis." |
| Look up a function | "What parameters does `insitupy.pp.normalize_total` accept? Show the full signature and docstring." |
| Explore plotting options | "List all spatial plotting functions available in InSituPy with a short description of each." |
| Find implementation details | "Search the InSituPy codebase for how differential gene expression is implemented." |
| Understand storage | "Explain the on-disk directory structure of an InSituPy project." |
| Work with sample data | "How do I download and use the built-in example datasets?" |
| SpatialData interop | "How do I convert an `InSituData` object to SpatialData format?" |
| Interactive visualization | "What does `data.show()` open and how does interactive annotation work?" |
| Multi-sample analysis | "Give me an example of loading multiple Xenium slides into an `InSituExperiment` and running cross-sample DGE." |

---

## Tool Reference

The server exposes the following 22 tools. Your AI assistant selects the appropriate one automatically.

### Generic Introspection

| Tool | What it does |
|------|-------------|
| `list_modules` | List all InSituPy submodules / subpackages |
| `list_classes` | List all classes defined in a module |
| `list_functions` | List all public functions with signatures |
| `get_class_info` | Full docstring, `__init__` signature, methods, and properties of a class |
| `get_function_source` | Full source code of a function, method, or class |
| `get_docstring` | Complete docstring for any module, class, or function |
| `search_codebase` | Regex search across all source and test files |
| `list_test_files` | List all test files with descriptions |
| `read_source_file` | Read a file by path or dotted module path with line numbers |

### InSituPy-Specific Overviews

| Tool | What it does |
|------|-------------|
| `get_data_model` | Hierarchical data model (`InSituData`, `InSituExperiment`, etc.) |
| `get_io_formats` | Supported input/output formats and reader functions |
| `get_plotting_api` | All plotting functions organized by category |
| `get_preprocessing_api` | All preprocessing functions |
| `get_tools_api` | Analysis tools (DGE, distance, neighbors, pseudobulk, registration) |
| `get_public_api` | Top-level exports from `import insitupy` |
| `get_workflow_guide` | Nine common workflows with runnable code examples |
| `get_storage_format` | On-disk project directory structure and metadata schema |
| `get_datasets_guide` | Available sample datasets and how to download them |
| `get_result_types` | Result objects returned by analysis tools |
| `get_interactive_guide` | napari-based interactive visualization |
| `get_images_api` | Image I/O and utility functions |
| `get_spatialdata_api` | SpatialData conversion functions |

---

## Local Development Setup

> **This section is for advanced users only.** Most users should use the `uvx` one-liner described above.

If you want to run the server from a local clone of the repository — for example, to test unreleased changes — follow these steps:

**1. Clone the repository and install dependencies:**

```bash
git clone https://github.com/SpatialPathology/InSituPy.git
cd InSituPy
pip install "insitupy-spatial[mcp]"
```

**2. Configure your client** using the local Python interpreter that has `insitupy-spatial` installed. Replace the paths below with your actual paths:

```json
{
  "mcpServers": {
    "insitupy": {
      "command": "/path/to/your/python",
      "args": ["-m", "tools.mcp_server"]
    }
  }
}
```

On macOS with a conda environment:

```json
{
  "mcpServers": {
    "insitupy": {
      "command": "/Users/yourname/miniforge3/envs/insitupy/bin/python",
      "args": ["-m", "tools.mcp_server"]
    }
  }
}
```

On Windows:

```json
{
  "mcpServers": {
    "insitupy": {
      "command": "C:\\Users\\yourname\\miniforge3\\envs\\insitupy\\python.exe",
      "args": ["-m", "tools.mcp_server"]
    }
  }
}
```

> **Note:** The `python` interpreter must belong to the environment where `insitupy-spatial` is installed, and the working directory must be the repository root (so `tools.mcp_server` is importable).
