# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MCP server for inspecting and operating a running Geti instance over its REST API."""

from geti_mcp.config import ServerConfig, config_from_args
from geti_mcp.errors import ErrorCode, GetiMcpError
from geti_mcp.server import build_server

__all__ = ["ErrorCode", "GetiMcpError", "ServerConfig", "build_server", "config_from_args"]
