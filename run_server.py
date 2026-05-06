"""
Launcher for the InSituPy MCP server.

Explicitly adds the repo root to sys.path so that the `tools` package is
importable regardless of the working directory. Use this script when running
from Claude Desktop (which ignores the `cwd` config field).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.mcp_server.server import mcp

mcp.run(transport="stdio")
