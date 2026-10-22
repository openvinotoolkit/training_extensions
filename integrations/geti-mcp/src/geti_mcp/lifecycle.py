# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Normalized job lifecycle.

Geti reports ``JobView.status`` as the name of an internal ``IntEnum``. This module maps
those names onto a small documented lifecycle while always preserving the original string,
so a backend that gains a new status degrades to ``unknown`` instead of being misreported.
"""

from __future__ import annotations

from enum import StrEnum


class NormalizedJobState(StrEnum):
    """Documented lifecycle exposed by this adapter.

    Attributes:
        QUEUED: Accepted by Geti, not yet started.
        RUNNING: Actively executing.
        CANCELLING: Cancellation was requested; the job has not stopped yet.
        SUCCEEDED: Terminal, completed successfully.
        FAILED: Terminal, ended with an error.
        CANCELLED: Terminal, stopped after a cancellation request.
        UNKNOWN: Geti reported a status this adapter does not recognize.
    """

    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


#: Geti ``JobStatus`` names, verified against ``app/core/jobs/models/job.py``.
_STATUS_MAP: dict[str, NormalizedJobState] = {
    "PENDING": NormalizedJobState.QUEUED,
    "RUNNING": NormalizedJobState.RUNNING,
    "CANCELLING": NormalizedJobState.CANCELLING,
    "DONE": NormalizedJobState.SUCCEEDED,
    "FAILED": NormalizedJobState.FAILED,
    "CANCELLED": NormalizedJobState.CANCELLED,
}

TERMINAL_STATES = frozenset({NormalizedJobState.SUCCEEDED, NormalizedJobState.FAILED, NormalizedJobState.CANCELLED})


def normalize_status(backend_status: str | None) -> NormalizedJobState:
    """Map a Geti job status name onto the normalized lifecycle.

    Args:
        backend_status: The raw ``JobView.status`` string.

    Returns:
        NormalizedJobState: The normalized state; ``UNKNOWN`` for unrecognized input.
    """
    if not backend_status:
        return NormalizedJobState.UNKNOWN
    return _STATUS_MAP.get(backend_status.strip().upper(), NormalizedJobState.UNKNOWN)


def is_terminal(state: NormalizedJobState) -> bool:
    """Whether a normalized state means the job will not change again.

    ``CANCELLING`` is deliberately not terminal: cancellation has been requested but the
    job may still be running, and may still succeed or fail.

    Args:
        state: The normalized state.

    Returns:
        bool: ``True`` only for ``succeeded``, ``failed``, or ``cancelled``.
    """
    return state in TERMINAL_STATES
