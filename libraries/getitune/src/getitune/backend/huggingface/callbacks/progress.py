# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Progress reporting for the Hugging Face backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from transformers import TrainerCallback

if TYPE_CHECKING:
    from collections.abc import Callable

    from transformers import TrainerControl, TrainerState, TrainingArguments

__all__ = ["HFProgressCallback", "extract_progress_fn"]


def extract_progress_fn(
    callbacks: list[Any] | None,
) -> tuple[Callable[[float], None] | None, float, float]:
    """Find the application's progress-reporting callback, if any.

    Scans for an object with ``_on_progress_update``, ``_min_p``, and ``_max_p``
    attributes, duck-typed so this module doesn't have to import the
    application's ``TrainingProgressCallback``. Everything else in
    *callbacks* — including genuine ``transformers.TrainerCallback``
    instances — is left for the caller to pass to ``Trainer`` directly; this
    function only extracts the progress hook.

    Args:
        callbacks: Callback objects passed to ``HFEngine.train()``, or
            ``None``.

    Returns:
        ``(progress_fn, min_p, max_p)``, or ``(None, 0.0, 100.0)`` if no
        callback in the list looks like a progress reporter.
    """
    if not callbacks:
        return None, 0.0, 100.0

    for callback in callbacks:
        fn = getattr(callback, "_on_progress_update", None)
        if fn is not None:
            return fn, getattr(callback, "_min_p", 0.0), getattr(callback, "_max_p", 100.0)

    return None, 0.0, 100.0


class HFProgressCallback(TrainerCallback):
    """Reports training progress through a duck-typed callable.

    Interpolates ``state.global_step / state.max_steps`` into
    ``[min_p, max_p]`` and calls *progress_fn* with the result on every
    logged step.
    """

    def __init__(self, progress_fn: Callable[[float], None], min_p: float = 0.0, max_p: float = 100.0) -> None:
        self._progress_fn = progress_fn
        self._min_p = min_p
        self._max_p = max_p

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Report progress whenever the ``Trainer`` logs, i.e. at ``logging_steps``."""
        if state.max_steps <= 0:
            return
        fraction = min(max(state.global_step / state.max_steps, 0.0), 1.0)
        self._progress_fn(self._min_p + fraction * (self._max_p - self._min_p))
