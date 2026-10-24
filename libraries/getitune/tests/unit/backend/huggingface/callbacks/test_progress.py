# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for progress reporting (G18)."""

from __future__ import annotations

from types import SimpleNamespace

from getitune.backend.huggingface.callbacks.progress import HFProgressCallback, extract_progress_fn


class _Progress:
    """Stand-in for the application's TrainingProgressCallback (duck-typed)."""

    def __init__(self, min_p: float = 0.0, max_p: float = 100.0) -> None:
        self._min_p = min_p
        self._max_p = max_p
        self.calls: list[float] = []

    def _on_progress_update(self, value: float) -> None:
        self.calls.append(value)


class TestExtractProgressFn:
    def test_returns_none_for_empty_list(self) -> None:
        fn, min_p, max_p = extract_progress_fn([])
        assert fn is None
        assert (min_p, max_p) == (0.0, 100.0)

    def test_returns_none_for_none(self) -> None:
        fn, min_p, max_p = extract_progress_fn(None)
        assert fn is None

    def test_finds_the_duck_typed_progress_object(self) -> None:
        progress = _Progress(min_p=10.0, max_p=90.0)
        fn, min_p, max_p = extract_progress_fn([object(), progress])
        assert fn is not None
        fn(42.0)
        assert progress.calls == [42.0]
        assert (min_p, max_p) == (10.0, 90.0)

    def test_ignores_objects_without_the_progress_attribute(self) -> None:
        fn, _, _ = extract_progress_fn([object(), SimpleNamespace(unrelated=True)])
        assert fn is None


class TestHFProgressCallback:
    def test_interpolates_step_fraction_into_min_max_range(self) -> None:
        progress = _Progress()
        callback = HFProgressCallback(progress._on_progress_update, min_p=0.0, max_p=100.0)
        state = SimpleNamespace(global_step=5, max_steps=10)

        callback.on_log(args=None, state=state, control=None)  # type: ignore[arg-type]

        assert progress.calls == [50.0]

    def test_respects_a_narrower_progress_range(self) -> None:
        progress = _Progress()
        callback = HFProgressCallback(progress._on_progress_update, min_p=20.0, max_p=40.0)
        state = SimpleNamespace(global_step=5, max_steps=10)

        callback.on_log(args=None, state=state, control=None)  # type: ignore[arg-type]

        assert progress.calls == [30.0]

    def test_reports_100_percent_at_the_last_step(self) -> None:
        progress = _Progress()
        callback = HFProgressCallback(progress._on_progress_update)
        state = SimpleNamespace(global_step=10, max_steps=10)

        callback.on_log(args=None, state=state, control=None)  # type: ignore[arg-type]

        assert progress.calls == [100.0]

    def test_does_nothing_when_max_steps_is_unknown(self) -> None:
        progress = _Progress()
        callback = HFProgressCallback(progress._on_progress_update)
        state = SimpleNamespace(global_step=0, max_steps=0)

        callback.on_log(args=None, state=state, control=None)  # type: ignore[arg-type]

        assert progress.calls == []
