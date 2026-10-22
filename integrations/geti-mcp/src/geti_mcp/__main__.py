# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Process entry point for the stdio MCP server."""

from __future__ import annotations

import asyncio
import logging
import sys

from geti_mcp.config import ConfigError, ServerConfig, config_from_args
from geti_mcp.server import build_server, registered_tool_names


def configure_logging(level: str) -> None:
    """Send all logging to stderr.

    Stdout carries the MCP protocol framing, so a single stray write there corrupts the
    session. Every handler is replaced to catch libraries that default to stdout.

    Args:
        level: Log level name.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(level)


async def _serve(config: ServerConfig) -> None:
    server, client = build_server(config)
    logger = logging.getLogger("geti_mcp")
    logger.info(
        "Serving Geti MCP over stdio for %s; scope=%s; tools=%s",
        config.base_url,
        "all_projects" if config.all_projects_allowed else f"{len(config.allowed_projects or ())} project(s)",
        ", ".join(registered_tool_names(config)),
    )
    try:
        await server.run_stdio_async()
    finally:
        await client.aclose()


def main(argv: list[str] | None = None) -> int:
    """Run the stdio MCP server.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        int: Process exit code.
    """
    try:
        config = config_from_args(argv)
    except ConfigError as exc:
        print(f"geti-mcp: {exc}", file=sys.stderr)
        return 2

    configure_logging(config.log_level)
    try:
        asyncio.run(_serve(config))
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
