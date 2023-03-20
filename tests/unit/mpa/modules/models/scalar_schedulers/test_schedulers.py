"""Test schedulers."""


# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from otx.algorithms.segmentation.adapters.mmseg.models.schedulers import (
    ConstantScalarScheduler,
    PolyScalarScheduler,
    StepScalarScheduler,
)


class TestSchedulers:
    """Test schedulers."""

    def test_constant_scalar_scheduler(self):
        """Test constant scalar scheduler.

        Learning rate should not change over time.
        """
        scheduler = ConstantScalarScheduler(scale=30.0)
        assert scheduler(0, 1) == 30.0
        assert scheduler(1, 1) == 30.0
        assert scheduler(2, 10) == 30.0

    def test_constant_scalar_scheduler_invalid_scale(self):
        """Test constant scalar scheduler with invalid scale."""
        with pytest.raises(AssertionError):
            ConstantScalarScheduler(scale=-1.0)

    @pytest.mark.xfail
    def test_constant_scalar_scheduler_invalid_step(self):
        """Test constant scalar scheduler with invalid step.

        TODO: ConstantScalarScheculer should be modified to raise this error
        """
        scheduler = ConstantScalarScheduler(scale=30.0)
        with pytest.raises(AssertionError):
            scheduler(-1, 1)

    def test_poly_scalar_scheduler_by_epoch_false(self):
        """Test poly scalar scheduler."""
        # By epoch is False
        scheduler = PolyScalarScheduler(
            start_scale=30.0,
            end_scale=0.0,
            num_iters=100,
            power=0.9,
            by_epoch=False,
        )

        # learning rate should decrease over time
        assert scheduler(0, 1) == 30.0
        assert scheduler(1, 1) < 30.0
        assert scheduler(2, 1) < scheduler(1, 1)
        assert scheduler(3, 1) < scheduler(2, 1)

        assert scheduler(50, 10) == scheduler(50, 1)  # as this is not by epoch

        # learning rate should not change after num_iters
        assert scheduler(100, 1) == 0.0
        assert scheduler(101, 1) == 0.0
        assert scheduler(102, 1) == 0.0

    def test_poly_scalar_scheduler_by_epoch_true(self):
        scheduler = PolyScalarScheduler(
            start_scale=30.0,
            end_scale=0.0,
            num_iters=100,
            power=0.9,
            by_epoch=True,
        )

        # learning rate should decrease over time
        assert scheduler(0, 1) == 30.0
        assert scheduler(1, 1) < 30.0
        assert scheduler(2, 1) < scheduler(1, 1)
        assert scheduler(3, 1) < scheduler(2, 1)

        assert scheduler(50, 10) != scheduler(50, 1)  # as this is by epoch

        # learning rate should not change after num_iters
        assert scheduler(100, 1) == 0.0
        assert scheduler(101, 1) == 0.0
        assert scheduler(102, 1) == 0.0

    def test_step_scalar_scheduler_by_epoch_false(self):
        """Test step scalar scheduler."""
        # By epoch is False
        scheduler = StepScalarScheduler(
            scales=[30.0, 20.0, 10.0, 5.0],
            num_iters=[2, 3, 4],
            by_epoch=False,
        )

        # learning rate should decrease over time as a step function
        assert scheduler(0, 1) == 30.0
        assert scheduler(1, 1) == 30.0
        assert scheduler(2, 1) < scheduler(1, 1)
        assert scheduler(3, 1) < scheduler(2, 1)

        assert scheduler(50, 10) == scheduler(50, 1)

        assert scheduler(5, 2) == 5.0
        assert scheduler(5, 0) == scheduler(10, 1)

        assert scheduler(10, 1) == 5.0  # steps greater than total num_iters

    def test_step_scalar_scheduler_by_epoch_true(self):
        # By epoch is True
        scheduler = StepScalarScheduler(
            scales=[30.0, 20.0, 10.0, 5.0],
            num_iters=[2, 3, 4],
            by_epoch=True,
        )

        # learning rate should decrease over time as a step function
        assert scheduler(0, 1) == 30.0
        assert scheduler(1, 1) == 30.0
        assert scheduler(2, 1) < scheduler(1, 1)
        assert scheduler(3, 1) < scheduler(2, 1)

        assert scheduler(9, 5) == 30.0
        assert scheduler(5, 2) == 20.0
        assert scheduler(5, 2) < scheduler(10, 11)
