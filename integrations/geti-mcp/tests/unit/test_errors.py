# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for error payloads and secret redaction."""

from __future__ import annotations

import json

import pytest

from geti_mcp.errors import MAX_ERROR_MESSAGE_CHARS, ErrorCode, GetiMcpError, redact


class TestToolError:
    def test_payload_is_machine_readable(self) -> None:
        error = GetiMcpError(
            ErrorCode.PERMISSION_DENIED,
            "Training is not enabled.",
            guidance="Ask the operator to restart with --allow-training.",
            details={"project_id": "abc"},
        )
        payload = json.loads(str(error.to_tool_error()))
        assert payload["error_code"] == "permission_denied"
        assert payload["message"] == "Training is not enabled."
        assert payload["guidance"].startswith("Ask the operator")
        assert payload["details"] == {"project_id": "abc"}

    def test_optional_fields_are_omitted(self) -> None:
        payload = json.loads(str(GetiMcpError(ErrorCode.NOT_FOUND, "Missing.").to_tool_error()))
        assert payload == {"error_code": "not_found", "message": "Missing."}

    def test_every_code_is_a_stable_snake_case_string(self) -> None:
        for code in ErrorCode:
            assert code.value == code.value.lower()
            assert " " not in code.value


class TestRedaction:
    @pytest.mark.parametrize(
        ("raw", "leaked"),
        [
            ("connect to https://admin:hunter2@geti.test failed", "hunter2"),
            ("api_key=sk-lives-here rejected", "sk-lives-here"),
            ("token: abcd1234efgh", "abcd1234efgh"),
            ("Authorization=Bearer zzzz9999", "zzzz9999"),
            ("password = p@ssw0rd!", "p@ssw0rd!"),
        ],
    )
    def test_credentials_are_removed(self, raw: str, leaked: str) -> None:
        assert leaked not in redact(raw)

    @pytest.mark.parametrize(
        "raw",
        [
            "failed reading /Users/alice/Desktop/Projects/geti/data/geti.db",
            r"failed reading C:\Users\alice\AppData\geti\geti.db",
            "no such file: /home/bob/.ssh/id_rsa",
        ],
    )
    def test_filesystem_paths_are_removed(self, raw: str) -> None:
        cleaned = redact(raw)
        assert "alice" not in cleaned
        assert "bob" not in cleaned

    def test_long_messages_are_truncated(self) -> None:
        cleaned = redact("x" * 5000)
        assert len(cleaned) <= MAX_ERROR_MESSAGE_CHARS + 1

    def test_whitespace_is_collapsed(self) -> None:
        assert redact("a\n\n   b\tc") == "a b c"

    def test_harmless_text_survives(self) -> None:
        assert redact("Project not found") == "Project not found"

    def test_empty_input_is_safe(self) -> None:
        assert redact("") == ""
