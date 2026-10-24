# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import lightning.pytorch as pl
import pytest
import torch
from lightning.pytorch.trainer.states import TrainerFn

from getitune.backend.lightning.callbacks.adaptive_early_stopping import EarlyStoppingWithWarmup


class TestEarlyStoppingWithWarmup:
    """Test cases for EarlyStoppingWithWarmup callback."""

    @pytest.fixture
    def mock_trainer(self):
        """Create a mock trainer for testing."""
        trainer = MagicMock(spec=pl.Trainer)
        trainer.global_step = 0
        trainer.num_training_batches = 100
        trainer.check_val_every_n_epoch = 1
        trainer.current_epoch = 0
        trainer.sanity_checking = False
        trainer.fast_dev_run = False
        trainer.should_stop = False

        class MockState:
            fn = TrainerFn.FITTING

        trainer.state = MockState()
        return trainer

    @pytest.fixture
    def mock_pl_module(self):
        """Create a mock PyTorch Lightning module for testing."""
        return MagicMock(spec=pl.LightningModule)

    def test_init_default_values(self):
        """Test initialization with default values."""
        callback = EarlyStoppingWithWarmup(monitor="val_loss")

        assert callback.monitor == "val_loss"
        assert callback.min_delta == 0.0
        assert callback.patience == 10
        assert callback.verbose is False
        assert callback.mode == "min"
        assert callback.strict is True
        assert callback.check_finite is True
        assert callback.stopping_threshold is None
        assert callback._check_on_train_epoch_end is False
        assert callback.divergence_threshold is None
        assert callback.log_rank_zero_only is False
        assert callback.warmup_iters == 100
        assert callback.warmup_epochs == 3

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        callback = EarlyStoppingWithWarmup(
            monitor="val_acc",
            min_delta=0.001,
            patience=5,
            verbose=True,
            mode="max",
            strict=False,
            check_finite=False,
            stopping_threshold=0.95,
            divergence_threshold=0.1,
            check_on_train_epoch_end=True,
            log_rank_zero_only=True,
            warmup_iters=50,
            warmup_epochs=2,
        )

        assert callback.monitor == "val_acc"
        assert callback.min_delta == 0.001
        assert callback.patience == 5
        assert callback.verbose is True
        assert callback.mode == "max"
        assert callback.strict is False
        assert callback.check_finite is False
        assert callback.stopping_threshold == 0.95
        assert callback.divergence_threshold == 0.1
        assert callback._check_on_train_epoch_end is True
        assert callback.log_rank_zero_only is True
        assert callback.warmup_iters == 50
        assert callback.warmup_epochs == 2

    def test_should_skip_check_warmup_epochs(self, mock_trainer):
        """Test that early stopping is skipped during warmup epochs."""
        callback = EarlyStoppingWithWarmup(
            monitor="val_loss",
            warmup_iters=50,
            warmup_epochs=3,
        )

        # Set trainer parameters
        mock_trainer.num_training_batches = 100
        mock_trainer.global_step = 200  # 2 epochs worth of steps

        # Warmup threshold = max(3 * 100, 50) = 300
        # Since global_step (200) < warmup_threshold (300), should skip
        assert callback._should_skip_check(mock_trainer) is True

    def test_should_skip_check_warmup_iters(self, mock_trainer):
        """Test that early stopping is skipped during warmup iterations."""
        callback = EarlyStoppingWithWarmup(
            monitor="val_loss",
            warmup_iters=500,
            warmup_epochs=1,
        )

        # Set trainer parameters
        mock_trainer.num_training_batches = 100
        mock_trainer.global_step = 200

        # Warmup threshold = max(1 * 100, 500) = 500
        # Since global_step (200) < warmup_threshold (500), should skip
        assert callback._should_skip_check(mock_trainer) is True

    def test_should_not_skip_check_after_warmup(self, mock_trainer):
        """Test that early stopping is not skipped after warmup period."""
        callback = EarlyStoppingWithWarmup(
            monitor="val_loss",
            warmup_iters=100,
            warmup_epochs=2,
        )

        # Set trainer parameters
        mock_trainer.num_training_batches = 100
        mock_trainer.global_step = 400

        # Mock the parent's _should_skip_check to return False
        with pytest.MonkeyPatch().context() as m:
            m.setattr(
                "lightning.pytorch.callbacks.early_stopping.EarlyStopping._should_skip_check",
                lambda _, __: False,
            )

            # Warmup threshold = max(2 * 100, 100) = 200
            # Since global_step (400) >= warmup_threshold (200), should not skip
            assert callback._should_skip_check(mock_trainer) is False

    def test_warmup_threshold_calculation(self, mock_trainer):
        """Test warmup threshold calculation uses max of warmup_epochs * batches and warmup_iters."""
        callback = EarlyStoppingWithWarmup(
            monitor="val_loss",
            warmup_iters=50,
            warmup_epochs=3,
        )

        mock_trainer.num_training_batches = 100

        # Test case 1: warmup_epochs * batches > warmup_iters
        mock_trainer.global_step = 200
        # Warmup threshold = max(3 * 100, 50) = 300
        assert callback._should_skip_check(mock_trainer) is True

        mock_trainer.global_step = 350
        assert callback._should_skip_check(mock_trainer) is False

        # Test case 2: warmup_iters > warmup_epochs * batches
        callback.warmup_iters = 500
        callback.warmup_epochs = 1

        mock_trainer.global_step = 200
        # Warmup threshold = max(1 * 100, 500) = 500
        assert callback._should_skip_check(mock_trainer) is True

        mock_trainer.global_step = 550
        with pytest.MonkeyPatch().context() as m:
            m.setattr(
                "lightning.pytorch.callbacks.early_stopping.EarlyStopping._should_skip_check",
                lambda _, __: False,
            )
            assert callback._should_skip_check(mock_trainer) is False

    def test_zero_warmup_values(self, mock_trainer):
        """Test behavior with zero warmup values."""
        callback = EarlyStoppingWithWarmup(
            monitor="val_loss",
            warmup_iters=0,
            warmup_epochs=0,
        )

        mock_trainer.num_training_batches = 100
        mock_trainer.global_step = 1

        # Warmup threshold = max(0 * 100, 0) = 0
        # Since global_step (1) >= warmup_threshold (0), should not skip due to warmup
        with pytest.MonkeyPatch().context() as m:
            m.setattr(
                "lightning.pytorch.callbacks.early_stopping.EarlyStopping._should_skip_check",
                lambda _, __: False,
            )
            assert callback._should_skip_check(mock_trainer) is False

    def test_check_on_train_epoch_end_false_behavior(self, mock_trainer, mock_pl_module):
        """Test that when check_on_train_epoch_end=False, checks only happen on validation_end."""
        callback = EarlyStoppingWithWarmup(
            monitor="val_loss",
            patience=2,
            warmup_iters=10,
            warmup_epochs=1,
            check_on_train_epoch_end=False,
        )

        # Setup past warmup
        mock_trainer.num_training_batches = 100
        mock_trainer.global_step = 200
        mock_trainer.current_epoch = 3
        mock_trainer.callback_metrics = {"val_loss": torch.tensor(0.8)}  # Poor performance

        callback.best_score = torch.tensor(0.1)
        mock_trainer.callback_metrics = {"val_loss": torch.tensor(0.7)}
        callback.on_validation_end(mock_trainer, mock_pl_module)
        assert callback.wait_count == 1

        callback.on_train_epoch_end(mock_trainer, mock_pl_module)
        assert callback.wait_count == 1

    def test_check_on_train_epoch_end_true_behavior(self, mock_trainer, mock_pl_module):
        """Test that when check_on_train_epoch_end=True, checks happen on both train and validation end."""
        callback = EarlyStoppingWithWarmup(
            monitor="val_loss",  # Monitor training loss for this test
            patience=2,
            warmup_iters=10,
            warmup_epochs=1,
            check_on_train_epoch_end=True,
        )

        # Setup past warmup
        mock_trainer.num_training_batches = 100
        mock_trainer.global_step = 200
        mock_trainer.current_epoch = 3
        mock_trainer.callback_metrics = {"val_loss": torch.tensor(0.8)}

        callback.best_score = torch.tensor(0.1)
        mock_trainer.callback_metrics = {"val_loss": torch.tensor(0.7)}
        callback.on_validation_end(mock_trainer, mock_pl_module)
        assert callback.wait_count == 0

        callback.on_train_epoch_end(mock_trainer, mock_pl_module)
        assert callback.wait_count == 1

    def test_warmup_prevents_early_termination(self, mock_trainer, mock_pl_module):
        """Test that early stopping is prevented during warmup even with poor metrics."""
        callback = EarlyStoppingWithWarmup(
            monitor="val_loss",
            patience=1,  # Very low patience
            min_delta=0.0,
            warmup_iters=200,
            warmup_epochs=3,
            check_on_train_epoch_end=False,
        )

        mock_trainer.num_training_batches = 50
        mock_trainer.should_stop = False

        # During warmup with very poor validation loss
        mock_trainer.global_step = 100  # Less than warmup threshold (max(3*50, 200) = 200)
        mock_trainer.current_epoch = 2
        mock_trainer.callback_metrics = {"val_loss": 10.0}  # Very high loss

        # Should skip check due to warmup, preventing early stopping
        assert callback._should_skip_check(mock_trainer) is True

        # Even after multiple calls during warmup, should not stop
        for _ in range(5):
            callback.on_validation_end(mock_trainer, mock_pl_module)

        # After warmup period
        mock_trainer.global_step = 250  # Greater than warmup threshold
        mock_trainer.current_epoch = 5

        # Now should not skip due to warmup
        with pytest.MonkeyPatch().context() as m:
            m.setattr(
                "lightning.pytorch.callbacks.early_stopping.EarlyStopping._should_skip_check",
                lambda _, __: False,
            )
            assert callback._should_skip_check(mock_trainer) is False

    def test_warmup_calculation_edge_cases(self, mock_trainer):
        """Test edge cases in warmup threshold calculation."""
        # Edge case 1: Very small warmup_epochs with large batch size
        callback = EarlyStoppingWithWarmup(
            monitor="val_loss",
            warmup_iters=1,
            warmup_epochs=0,
        )

        mock_trainer.num_training_batches = 1000
        mock_trainer.global_step = 0

        # Warmup threshold = max(0 * 1000, 1) = 1
        assert callback._should_skip_check(mock_trainer) is True

        mock_trainer.global_step = 1
        with pytest.MonkeyPatch().context() as m:
            m.setattr(
                "lightning.pytorch.callbacks.early_stopping.EarlyStopping._should_skip_check",
                lambda _, __: False,
            )
            assert callback._should_skip_check(mock_trainer) is False

        # Edge case 2: Large warmup_epochs with very small batch size
        callback = EarlyStoppingWithWarmup(
            monitor="val_loss",
            warmup_iters=1,
            warmup_epochs=1000,
        )

        mock_trainer.num_training_batches = 1
        mock_trainer.global_step = 500

        # Warmup threshold = max(1000 * 1, 1) = 1000
        assert callback._should_skip_check(mock_trainer) is True

        mock_trainer.global_step = 1001
        with pytest.MonkeyPatch().context() as m:
            m.setattr(
                "lightning.pytorch.callbacks.early_stopping.EarlyStopping._should_skip_check",
                lambda _, __: False,
            )
            assert callback._should_skip_check(mock_trainer) is False
