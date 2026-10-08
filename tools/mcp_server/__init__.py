"""InSituPy MCP server for API introspection and codebase exploration."""


def main() -> None:
    """Entry point for the ``insitupy-mcp`` console script."""
    from .server import mcp

    mcp.run(transport="stdio")
