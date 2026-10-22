# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the normalized job lifecycle."""

from __future__ import annotations

import pytest

from geti_mcp.lifecycle import TERMINAL_STATES, NormalizedJobState, is_terminal, normalize_status


@pytest.mark.parametrize(
    ("backend_status", "expected"),
    [
        ("PENDING", NormalizedJobState.QUEUED),
        ("RUNNING", NormalizedJobState.RUNNING),
        ("CANCELLING", NormalizedJobState.CANCELLING),
        ("DONE", NormalizedJobState.SUCCEEDED),
        ("FAILED", NormalizedJobState.FAILED),
        ("CANCELLED", NormalizedJobState.CANCELLED),
    ],
)
def test_every_backend_status_maps_to_a_distinct_state(backend_status: str, expected: NormalizedJobState) -> None:
    assert normalize_status(backend_status) is expected


@pytest.mark.parametrize("raw", ["done", "Done", " DONE "])
def test_mapping_tolerates_case_and_padding(raw: str) -> None:
    assert normalize_status(raw) is NormalizedJobState.SUCCEEDED


@pytest.mark.parametrize("raw", ["PAUSED", "", None, "SUCCESS"])
def test_unrecognized_status_is_reported_as_unknown_not_guessed(raw: str | None) -> None:
    assert normalize_status(raw) is NormalizedJobState.UNKNOWN


def test_cancellation_requested_is_not_terminal() -> None:
    """A cancellation request is not proof the job stopped."""
    assert NormalizedJobState.CANCELLING not in TERMINAL_STATES
    assert is_terminal(NormalizedJobState.CANCELLING) is False
    assert is_terminal(NormalizedJobState.CANCELLED) is True


def test_unknown_is_not_treated_as_terminal() -> None:
    assert is_terminal(NormalizedJobState.UNKNOWN) is False


def test_terminal_states_are_exactly_the_three_outcomes() -> None:
    assert {
        NormalizedJobState.SUCCEEDED,
        NormalizedJobState.FAILED,
        NormalizedJobState.CANCELLED,
    } == TERMINAL_STATES
