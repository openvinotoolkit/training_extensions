# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for operator configuration parsing and validation."""

from __future__ import annotations

import pytest

from geti_mcp.config import (
    BACKEND_MAX_PAGE_SIZE,
    MAX_CONFIGURABLE_WAIT_SECONDS,
    ConfigError,
    config_from_args,
    validate_base_url,
)

BASE = ["--base-url", "https://localhost:7860"]


class TestBaseUrl:
    @pytest.mark.parametrize(
        "raw",
        [
            "file:///etc/passwd",
            "ftp://geti.test",
            "geti.test:7860",
            "https://",
            "https://user:secret@geti.test",
            "https://geti.test/?token=abc",
            "https://geti.test/#fragment",
        ],
    )
    def test_rejects_unusable_or_credential_bearing_urls(self, raw: str) -> None:
        with pytest.raises(ConfigError):
            validate_base_url(raw)

    def test_normalizes_trailing_slash(self) -> None:
        assert validate_base_url("https://geti.test:7860/") == "https://geti.test:7860"

    def test_keeps_path_prefix(self) -> None:
        assert validate_base_url("https://geti.test/geti/") == "https://geti.test/geti"


class TestPermissions:
    def test_everything_is_denied_by_default(self) -> None:
        config = config_from_args([*BASE, "--all-projects"])
        assert config.permissions.allow_training is False
        assert config.permissions.allow_job_cancellation is False
        assert config.permissions.allow_image_access is False

    def test_opt_ins_are_independent(self) -> None:
        config = config_from_args([*BASE, "--all-projects", "--allow-training"])
        assert config.permissions.allow_training is True
        assert config.permissions.allow_job_cancellation is False
        assert config.permissions.allow_image_access is False


class TestProjectScope:
    def test_scope_must_be_explicit(self) -> None:
        with pytest.raises(ConfigError):
            config_from_args(BASE)

    def test_projects_and_all_projects_are_mutually_exclusive(self) -> None:
        with pytest.raises(SystemExit):
            config_from_args([*BASE, "--all-projects", "--projects", "7b073838-99d3-42ff-9018-4e901eb047fc"])

    def test_project_ids_are_canonicalized(self) -> None:
        config = config_from_args([*BASE, "--projects", "7B073838-99D3-42FF-9018-4E901EB047FC"])
        assert config.allowed_projects == frozenset({"7b073838-99d3-42ff-9018-4e901eb047fc"})
        assert config.all_projects_allowed is False

    def test_rejects_non_uuid_project(self) -> None:
        with pytest.raises(ConfigError):
            config_from_args([*BASE, "--projects", "animals"])

    def test_rejects_empty_project_list(self) -> None:
        with pytest.raises(ConfigError):
            config_from_args([*BASE, "--projects", " , "])


class TestLimits:
    def test_page_size_above_backend_maximum_is_rejected_loudly(self) -> None:
        """Silently clamping would hide an operator mistake, so startup fails instead."""
        with pytest.raises(ConfigError, match=str(BACKEND_MAX_PAGE_SIZE)):
            config_from_args([*BASE, "--all-projects", "--max-page-size", "5000"])

    def test_wait_above_the_cap_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match=str(MAX_CONFIGURABLE_WAIT_SECONDS)):
            config_from_args([*BASE, "--all-projects", "--max-wait-seconds", "9999"])

    def test_rejects_non_positive_values(self) -> None:
        with pytest.raises(ConfigError):
            config_from_args([*BASE, "--all-projects", "--max-items", "0"])

    def test_accepts_values_inside_the_allowed_range(self) -> None:
        config = config_from_args([*BASE, "--all-projects", "--max-page-size", "25", "--max-wait-seconds", "10"])
        assert config.limits.max_page_size == 25
        assert config.limits.max_wait_seconds == 10.0


class TestTls:
    def test_missing_ca_bundle_is_rejected_at_startup(self, tmp_path) -> None:
        missing = tmp_path / "nope.pem"
        with pytest.raises(ConfigError, match="does not exist"):
            config_from_args([*BASE, "--all-projects", "--ca-bundle", str(missing)])

    def test_directory_ca_bundle_is_rejected(self, tmp_path) -> None:
        with pytest.raises(ConfigError):
            config_from_args([*BASE, "--all-projects", "--ca-bundle", str(tmp_path)])

    def test_existing_ca_bundle_is_accepted(self, tmp_path) -> None:
        bundle = tmp_path / "geti.pem"
        bundle.write_text("-----BEGIN CERTIFICATE-----\n")
        config = config_from_args([*BASE, "--all-projects", "--ca-bundle", str(bundle)])
        assert config.ca_bundle == bundle.resolve()

    def test_no_option_disables_verification(self) -> None:
        """There is deliberately no flag that turns TLS verification off."""
        parser_options = {action.option_strings[0] for action in _actions()}
        assert not {"--insecure", "--no-verify", "--skip-tls-verify"} & parser_options


class TestEnvironment:
    def test_environment_provides_defaults(self, monkeypatch) -> None:
        monkeypatch.setenv("GETI_MCP_BASE_URL", "https://env.test:7860")
        monkeypatch.setenv("GETI_MCP_PROJECTS", "7b073838-99d3-42ff-9018-4e901eb047fc")
        monkeypatch.setenv("GETI_MCP_ALLOW_TRAINING", "true")
        config = config_from_args([])
        assert config.base_url == "https://env.test:7860"
        assert config.permissions.allow_training is True

    def test_command_line_overrides_environment(self, monkeypatch) -> None:
        monkeypatch.setenv("GETI_MCP_BASE_URL", "https://env.test:7860")
        config = config_from_args([*BASE, "--all-projects"])
        assert config.base_url == "https://localhost:7860"


def _actions() -> list:
    from geti_mcp.config import build_parser

    return [action for action in build_parser()._actions if action.option_strings]
