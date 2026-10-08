"""Allow running with ``python -m tools.mcp_server``."""

from .server import mcp

mcp.run(transport="stdio")
