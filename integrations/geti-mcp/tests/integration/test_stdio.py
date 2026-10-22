# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests over the real stdio transport.

These launch the installed ``geti-mcp`` executable as a subprocess and speak MCP to it,
against a throwaway HTTP server that replays the same fake Geti used by the unit tests.
No real Geti instance is required.
"""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
import pytest
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

from tests.conftest import ARCHITECTURE, JOB_A, PROJECT_A, PROJECT_B, FakeGeti

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _handler_factory(fake: FakeGeti) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            """Keep the test output clean."""

        def _dispatch(self, method: str) -> None:
            length = int(self.headers.get("content-length") or 0)
            body = self.rfile.read(length) if length else b""
            request = httpx.Request(method, f"http://127.0.0.1{self.path}", content=body)
            response = fake._handle(request)
            payload = response.read()
            self.send_response(response.status_code)
            self.send_header("content-type", response.headers.get("content-type", "application/json"))
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler naming
            self._dispatch("GET")

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler naming
            self._dispatch("POST")

    return Handler


@pytest.fixture
def geti_url(fake_geti: FakeGeti) -> Iterator[str]:
    """Serve the fake Geti over real HTTP on an ephemeral port."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_factory(fake_geti))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _params(geti_url: str, *extra: str) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "geti_mcp", "--base-url", geti_url, "--projects", PROJECT_A, *extra],
        env={"PATH": "/usr/bin:/bin"},
    )


async def test_initializes_and_advertises_read_only_tools(geti_url: str) -> None:
    async with stdio_client(_params(geti_url)) as (read, write), ClientSession(read, write) as session:
        init = await session.initialize()
        assert init.server_info.name == "geti"
        assert init.instructions

        names = {tool.name for tool in (await session.list_tools()).tools}
        assert "list_projects" in names
        assert {"start_training", "cancel_job", "view_media"} & names == set()


async def test_opt_ins_add_exactly_the_matching_tools(geti_url: str) -> None:
    async with (
        stdio_client(_params(geti_url, "--allow-training", "--allow-job-cancellation")) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        names = {tool.name for tool in (await session.list_tools()).tools}
        assert {"start_training", "cancel_job"} <= names
        assert "view_media" not in names


async def test_every_tool_publishes_input_and_output_schemas(geti_url: str) -> None:
    async with stdio_client(_params(geti_url)) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        for tool in (await session.list_tools()).tools:
            assert tool.input_schema.get("type") == "object", tool.name
            assert tool.output_schema, tool.name
            assert tool.description, tool.name


async def test_tool_call_returns_validated_structured_content(geti_url: str) -> None:
    async with stdio_client(_params(geti_url)) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        result = await session.call_tool("list_projects", {})
        assert result.is_error is False
        assert result.structured_content is not None
        assert [project["id"] for project in result.structured_content["projects"]] == [PROJECT_A]
        assert isinstance(result.content[0], types.TextContent)


async def test_unauthorized_call_is_denied_with_a_stable_error_code(geti_url: str) -> None:
    async with stdio_client(_params(geti_url)) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        result = await session.call_tool("get_project", {"project_id": PROJECT_B})
        assert result.is_error is True
        assert _error_payload(result)["error_code"] == "permission_denied"


async def test_a_disabled_tool_cannot_be_invoked_directly(geti_url: str) -> None:
    """The tool is not advertised, and calling it anyway still fails."""
    async with stdio_client(_params(geti_url)) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        result = await session.call_tool("start_training", {"project_id": PROJECT_A})
        assert result.is_error is True


async def test_enabled_training_tool_enforces_authorization_per_call(geti_url: str) -> None:
    async with (
        stdio_client(_params(geti_url, "--allow-training")) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        denied = await session.call_tool(
            "start_training",
            {"project_id": PROJECT_B, "model_architecture_id": ARCHITECTURE, "device": "cpu"},
        )
        assert denied.is_error is True
        assert _error_payload(denied)["error_code"] == "permission_denied"

        allowed = await session.call_tool(
            "start_training",
            {"project_id": PROJECT_A, "model_architecture_id": ARCHITECTURE, "device": "cpu"},
        )
        assert allowed.is_error is False
        assert allowed.structured_content["job_id"] == JOB_A


async def test_invalid_arguments_are_rejected_by_schema_validation(geti_url: str) -> None:
    async with stdio_client(_params(geti_url)) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        result = await session.call_tool("list_media", {"project_id": PROJECT_A, "offset": -5})
        assert result.is_error is True


async def test_stdout_carries_only_protocol_messages(geti_url: str, tmp_path) -> None:
    """Any stray stdout write would corrupt the session, so logging must stay on stderr."""
    stderr_path = tmp_path / "stderr.log"
    with stderr_path.open("wb") as stderr:
        async with (
            stdio_client(_params(geti_url, "--log-level", "DEBUG"), errlog=stderr) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            await session.list_tools()
            result = await session.call_tool("get_connection_info", {})
            assert result.is_error is False

    captured = stderr_path.read_text()
    assert "Serving Geti MCP over stdio" in captured
    assert '"jsonrpc"' not in captured


def _error_payload(result: types.CallToolResult) -> dict[str, Any]:
    """Extract the structured error body from a failed tool call.

    MCP prefixes tool errors with ``Error executing tool <name>: ``, so the JSON body this
    server raises is the trailing object.
    """
    block = result.content[0]
    assert isinstance(block, types.TextContent)
    return json.loads(block.text[block.text.index("{") :])
