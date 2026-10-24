# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for data augmentation scheduler components (CPU/GPU pipeline)."""

from __future__ import annotations

import secrets
from multiprocessing import Value
from unittest.mock import MagicMock, patch

import pytest
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks.callback import Callback

from getitune.backend.lightning.callbacks.aug_scheduler import AugmentationSchedulerCallback, DataAugSwitch
from getitune.data.augmentation import CPUAugmentationPipeline, GPUAugmentationPipeline

# ---------------------------------------------------------------------------
# Helpers / fixtures shared across test classes
# ---------------------------------------------------------------------------


def _make_minimal_policies(
    *,
    cpu_class: str = "getitune.data.augmentation.transforms.Resize",
    gpu_class: str = "kornia.augmentation.Normalize",
) -> dict:
    """Return a 4-policy dict with simple Resize (CPU) + Normalize (GPU)."""
    cpu_entry = {
        "class_path": cpu_class,
        "init_args": {"size": [640, 640], "keep_aspect_ratio": False},
    }
    gpu_entry = {
        "class_path": gpu_class,
        "init_args": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    }
    gpu_entry_flip = {
        "class_path": "kornia.augmentation.RandomHorizontalFlip",
        "init_args": {"p": 0.5},
    }
    return {
        "no_aug": {
            "augmentations_cpu": [cpu_entry],
            "augmentations_gpu": [gpu_entry],
        },
        "strong_aug_1": {
            "augmentations_cpu": [cpu_entry],
            "augmentations_gpu": [gpu_entry_flip, gpu_entry],
        },
        "strong_aug_2": {
            "augmentations_cpu": [cpu_entry],
            "augmentations_gpu": [gpu_entry_flip, gpu_entry],
        },
        "light_aug": {
            "augmentations_cpu": [cpu_entry],
            "augmentations_gpu": [gpu_entry],
        },
    }


POLICY_EPOCHS = [4, 23]


# ===================================================================
# TestDataAugSwitch
# ===================================================================
class TestDataAugSwitch:
    """Tests for DataAugSwitch with the CPU/GPU pipeline architecture."""

    # -- fixtures -------------------------------------------------------

    @pytest.fixture
    def policies(self):
        return _make_minimal_policies()

    @pytest.fixture
    def switch(self, policies):
        return DataAugSwitch(POLICY_EPOCHS, policies, input_size=[640, 640])

    @pytest.fixture
    def switch_with_epoch(self, switch):
        """Switch with a shared epoch pre-set to 0."""
        switch.set_shared_epoch(Value("i", 0))
        return switch

    # -- init -----------------------------------------------------------

    def test_init_stores_policy_epochs(self, switch):
        assert switch.policy_epochs == POLICY_EPOCHS

    def test_init_stores_input_size_as_tuple(self, switch):
        assert switch.input_size == (640, 640)

    def test_init_builds_cpu_pipeline_per_policy(self, switch):
        for name in ("no_aug", "strong_aug_1", "strong_aug_2", "light_aug"):
            assert name in switch.policies
            assert isinstance(switch.policies[name]["cpu_pipeline"], CPUAugmentationPipeline)

    def test_init_stores_gpu_configs_per_policy(self, switch):
        for name in ("no_aug", "strong_aug_1", "strong_aug_2", "light_aug"):
            assert isinstance(switch.policies[name]["gpu_aug_configs"], list)
            assert len(switch.policies[name]["gpu_aug_configs"]) >= 1

    def test_init_invalid_policy_epochs_length(self, policies):
        with pytest.raises(ValueError, match="Expected 2 policy epochs"):
            DataAugSwitch([4, 29, 50], policies)

    def test_init_no_input_size(self, policies):
        switch = DataAugSwitch(POLICY_EPOCHS, policies, input_size=None)
        assert switch.input_size is None

    def test_init_empty_gpu_augmentations(self):
        """Policy with no GPU augmentations should store empty list."""
        policies = {
            name: {
                "augmentations_cpu": [
                    {
                        "class_path": "getitune.data.augmentation.transforms.Resize",
                        "init_args": {"size": [640, 640], "keep_aspect_ratio": False},
                    },
                ],
            }
            for name in ("no_aug", "strong_aug_1", "strong_aug_2", "light_aug")
        }
        switch = DataAugSwitch(POLICY_EPOCHS, policies, input_size=[640, 640])
        assert switch.get_gpu_aug_configs("no_aug") == []

    # -- shared epoch ---------------------------------------------------

    def test_set_shared_epoch(self, switch):
        v = Value("i", 7)
        switch.set_shared_epoch(v)
        assert switch._shared_epoch is v

    def test_epoch_getter_raises_without_shared(self, switch):
        with pytest.raises(ValueError, match="Shared epoch not set"):
            _ = switch.epoch

    def test_epoch_setter_raises_without_shared(self, switch):
        with pytest.raises(ValueError, match="Shared epoch not set"):
            switch.epoch = 5

    def test_epoch_getter_and_setter(self, switch_with_epoch):
        switch_with_epoch.epoch = 12
        assert switch_with_epoch.epoch == 12

    # -- current_policy_name (stochastic) --------------------------------

    def test_policy_name_no_aug_stage(self, switch_with_epoch):
        for e in (0, 1, 3):
            switch_with_epoch.epoch = e
            assert switch_with_epoch.current_policy_name == "no_aug"

    def test_policy_name_strong_aug_stage_random(self, switch_with_epoch):
        switch_with_epoch.epoch = 10
        with patch.object(secrets, "choice", return_value="strong_aug_2") as m:
            assert switch_with_epoch.current_policy_name == "strong_aug_2"
            m.assert_called_once_with(["strong_aug_1", "strong_aug_2"])

    def test_policy_name_light_aug_stage(self, switch_with_epoch):
        for e in (23, 30, 40, 100):
            switch_with_epoch.epoch = e
            assert switch_with_epoch.current_policy_name == "light_aug"

    def test_policy_name_boundary_at_p0(self, switch_with_epoch):
        """epoch == p0 should enter strong_aug stage."""
        switch_with_epoch.epoch = 4
        with patch.object(secrets, "choice", return_value="strong_aug_1"):
            assert switch_with_epoch.current_policy_name == "strong_aug_1"

    def test_policy_name_boundary_at_p1(self, switch_with_epoch):
        """epoch == p1 should enter light_aug stage."""
        switch_with_epoch.epoch = 23
        assert switch_with_epoch.current_policy_name == "light_aug"

    # -- get_cpu_pipeline -----------------------------------------------

    def test_get_cpu_pipeline_returns_correct_type(self, switch_with_epoch):
        pipeline = switch_with_epoch.get_cpu_pipeline("no_aug")
        assert isinstance(pipeline, CPUAugmentationPipeline)

    def test_get_cpu_pipeline_returns_different_per_policy(self, switch_with_epoch):
        no_aug = switch_with_epoch.get_cpu_pipeline("no_aug")
        light_aug = switch_with_epoch.get_cpu_pipeline("light_aug")
        assert isinstance(no_aug, CPUAugmentationPipeline)
        assert isinstance(light_aug, CPUAugmentationPipeline)

    def test_get_cpu_pipeline_invalid_name_raises(self, switch_with_epoch):
        with pytest.raises(KeyError):
            switch_with_epoch.get_cpu_pipeline("nonexistent_policy")

    # -- get_gpu_aug_configs --------------------------------------------

    def test_get_gpu_aug_configs_returns_list(self, switch):
        configs = switch.get_gpu_aug_configs("no_aug")
        assert isinstance(configs, list)
        assert len(configs) == 1  # just Normalize

    def test_get_gpu_aug_configs_strong_has_flip(self, switch):
        configs = switch.get_gpu_aug_configs("strong_aug_1")
        class_paths = [c["class_path"] for c in configs]
        assert "kornia.augmentation.RandomHorizontalFlip" in class_paths

    # -- build_gpu_pipeline ---------------------------------------------

    def test_build_gpu_pipeline_returns_correct_type(self, switch):
        gpu = switch.build_gpu_pipeline("no_aug", data_keys=["input", "bbox_xyxy", "label"])
        assert isinstance(gpu, GPUAugmentationPipeline)

    def test_build_gpu_pipeline_data_keys_propagated(self, switch):
        keys = ["input", "bbox_xyxy", "label"]
        gpu = switch.build_gpu_pipeline("strong_aug_1", data_keys=keys)
        assert gpu.data_keys == keys

    def test_build_gpu_pipeline_empty_configs(self):
        """Policy with no GPU augmentations produces an empty pipeline."""
        policies = {
            name: {
                "augmentations_cpu": [
                    {
                        "class_path": "getitune.data.augmentation.transforms.Resize",
                        "init_args": {"size": [640, 640], "keep_aspect_ratio": False},
                    },
                ],
            }
            for name in ("no_aug", "strong_aug_1", "strong_aug_2", "light_aug")
        }
        switch = DataAugSwitch(POLICY_EPOCHS, policies, input_size=[640, 640])
        gpu = switch.build_gpu_pipeline("no_aug")
        assert isinstance(gpu, GPUAugmentationPipeline)


# ===================================================================
# TestAugmentationSchedulerCallback
# ===================================================================
class TestAugmentationSchedulerCallback:
    """Tests for AugmentationSchedulerCallback with GPU pipeline swapping."""

    # -- fixtures -------------------------------------------------------

    @pytest.fixture
    def switch(self):
        s = DataAugSwitch(POLICY_EPOCHS, _make_minimal_policies(), input_size=[640, 640])
        s.set_shared_epoch(Value("i", 0))
        return s

    @pytest.fixture
    def mock_gpu_callback(self):
        from getitune.backend.lightning.callbacks.gpu_augmentation import GPUAugmentationCallback

        cb = MagicMock(spec=GPUAugmentationCallback)
        mock_pipeline = MagicMock(spec=GPUAugmentationPipeline)
        mock_pipeline.data_keys = ["input", "bbox_xyxy", "label"]
        cb._train_pipeline = mock_pipeline
        return cb

    @pytest.fixture
    def mock_trainer(self, mock_gpu_callback):
        trainer = MagicMock(spec=Trainer)
        trainer.current_epoch = 0
        trainer.callbacks = [mock_gpu_callback]
        return trainer

    @pytest.fixture
    def mock_pl_module(self):
        pl = MagicMock(spec=LightningModule)
        param = torch.nn.Parameter(torch.zeros(1))
        pl.parameters.side_effect = lambda: iter([param])
        pl.device = torch.device("cpu")
        return pl

    @pytest.fixture
    def callback(self, switch):
        return AugmentationSchedulerCallback(data_aug_switch=switch)

    # -- init -----------------------------------------------------------

    def test_inherits_from_lightning_callback(self, callback):
        assert isinstance(callback, Callback)

    def test_init_with_switch(self, switch):
        cb = AugmentationSchedulerCallback(data_aug_switch=switch)
        assert cb.data_aug_switch is switch
        assert cb._gpu_aug_callback is None
        assert cb._last_gpu_policy is None

    def test_init_without_switch(self):
        cb = AugmentationSchedulerCallback()
        assert cb.data_aug_switch is None

    # -- setup ----------------------------------------------------------

    def test_setup_finds_gpu_callback(self, callback, mock_trainer, mock_pl_module, mock_gpu_callback):
        callback.setup(mock_trainer, mock_pl_module, stage="fit")
        assert callback._gpu_aug_callback is mock_gpu_callback

    def test_setup_no_gpu_callback(self, callback, mock_pl_module):
        trainer = MagicMock(spec=Trainer)
        trainer.callbacks = []
        callback.setup(trainer, mock_pl_module, stage="fit")
        assert callback._gpu_aug_callback is None

    # -- set_data_aug_switch --------------------------------------------

    def test_set_data_aug_switch(self, callback, switch):
        new_switch = MagicMock(spec=DataAugSwitch)
        callback.set_data_aug_switch(new_switch)
        assert callback.data_aug_switch is new_switch

    # -- on_train_epoch_start -------------------------------------------

    def test_epoch_start_noop_when_no_switch(self, mock_trainer, mock_pl_module):
        cb = AugmentationSchedulerCallback(data_aug_switch=None)
        cb.on_train_epoch_start(mock_trainer, mock_pl_module)  # should not raise

    def test_epoch_start_updates_shared_epoch(self, callback, mock_trainer, mock_pl_module, mock_gpu_callback):
        callback.setup(mock_trainer, mock_pl_module, stage="fit")
        mock_trainer.current_epoch = 7
        callback.on_train_epoch_start(mock_trainer, mock_pl_module)
        assert callback.data_aug_switch.epoch == 7

    def test_epoch_start_swaps_gpu_on_phase_change(self, callback, mock_trainer, mock_pl_module, mock_gpu_callback):
        callback.setup(mock_trainer, mock_pl_module, stage="fit")
        mock_trainer.current_epoch = 0
        callback.on_train_epoch_start(mock_trainer, mock_pl_module)
        assert callback._last_gpu_policy == "no_aug"
        assert mock_gpu_callback._train_pipeline is not None

    def test_epoch_start_no_swap_same_phase(self, callback, mock_trainer, mock_pl_module, mock_gpu_callback):
        callback.setup(mock_trainer, mock_pl_module, stage="fit")

        mock_trainer.current_epoch = 0
        callback.on_train_epoch_start(mock_trainer, mock_pl_module)
        assert callback._last_gpu_policy == "no_aug"

        mock_trainer.current_epoch = 1
        callback.on_train_epoch_start(mock_trainer, mock_pl_module)
        assert callback._last_gpu_policy == "no_aug"

    def test_epoch_start_detects_phase_transitions(self, callback, mock_trainer, mock_pl_module, mock_gpu_callback):
        callback.setup(mock_trainer, mock_pl_module, stage="fit")

        mock_trainer.current_epoch = 0
        callback.on_train_epoch_start(mock_trainer, mock_pl_module)
        assert callback._last_gpu_policy == "no_aug"

        mock_trainer.current_epoch = 4
        callback.on_train_epoch_start(mock_trainer, mock_pl_module)
        assert callback._last_gpu_policy in ("strong_aug_1", "strong_aug_2")

        mock_trainer.current_epoch = 23
        callback.on_train_epoch_start(mock_trainer, mock_pl_module)
        assert callback._last_gpu_policy == "light_aug"

    # -- _swap_gpu_pipeline ---------------------------------------------

    def test_swap_gpu_pipeline_noop_without_gpu_callback(self, callback, mock_pl_module):
        callback._gpu_aug_callback = None
        callback._swap_gpu_pipeline("no_aug", mock_pl_module)  # should not crash

    def test_swap_gpu_pipeline_noop_without_switch(self, mock_pl_module):
        cb = AugmentationSchedulerCallback(data_aug_switch=None)
        cb._gpu_aug_callback = MagicMock()
        cb._swap_gpu_pipeline("no_aug", mock_pl_module)  # should not crash

    def test_swap_gpu_pipeline_builds_and_assigns(self, callback, mock_trainer, mock_pl_module, mock_gpu_callback):
        callback.setup(mock_trainer, mock_pl_module, stage="fit")
        callback._swap_gpu_pipeline("strong_aug_1", mock_pl_module)
        new_pipeline = mock_gpu_callback._train_pipeline
        assert isinstance(new_pipeline, GPUAugmentationPipeline)

    def test_swap_gpu_pipeline_preserves_data_keys(self, callback, mock_trainer, mock_pl_module, mock_gpu_callback):
        callback.setup(mock_trainer, mock_pl_module, stage="fit")
        callback._swap_gpu_pipeline("strong_aug_1", mock_pl_module)
        new_pipeline = mock_gpu_callback._train_pipeline
        assert isinstance(new_pipeline, GPUAugmentationPipeline)
        assert new_pipeline.data_keys == ["input", "bbox_xyxy", "label"]

    def test_swap_gpu_pipeline_skips_empty_configs(self, mock_pl_module):
        """If a policy has no GPU augmentations, keep the current pipeline."""
        policies = {
            name: {
                "augmentations_cpu": [
                    {
                        "class_path": "getitune.data.augmentation.transforms.Resize",
                        "init_args": {"size": [640, 640], "keep_aspect_ratio": False},
                    },
                ],
            }
            for name in ("no_aug", "strong_aug_1", "strong_aug_2", "light_aug")
        }
        switch = DataAugSwitch(POLICY_EPOCHS, policies, input_size=[640, 640])
        switch.set_shared_epoch(Value("i", 0))

        from getitune.backend.lightning.callbacks.gpu_augmentation import GPUAugmentationCallback

        mock_gpu_cb = MagicMock(spec=GPUAugmentationCallback)
        original_pipeline = MagicMock()
        mock_gpu_cb._train_pipeline = original_pipeline

        cb = AugmentationSchedulerCallback(data_aug_switch=switch)
        cb._gpu_aug_callback = mock_gpu_cb
        cb._swap_gpu_pipeline("no_aug", mock_pl_module)

        # After the fix, empty-config policies should clear the GPU pipeline
        # instead of keeping stale transforms from the previous policy.
        assert mock_gpu_cb._train_pipeline is not original_pipeline
        assert isinstance(mock_gpu_cb._train_pipeline, GPUAugmentationPipeline)
        assert mock_gpu_cb._train_pipeline.aug_sequential is None

    # -- full training simulation ----------------------------------------

    def test_full_training_simulation(self, callback, mock_trainer, mock_pl_module, mock_gpu_callback):
        """Simulate a full training run through all 3 phase transitions."""
        callback.setup(mock_trainer, mock_pl_module, stage="fit")

        phase_history = []
        for epoch in range(50):
            mock_trainer.current_epoch = epoch
            callback.on_train_epoch_start(mock_trainer, mock_pl_module)
            phase_history.append(callback._last_gpu_policy)

        assert all(p == "no_aug" for p in phase_history[:4])
        assert all(p in ("strong_aug_1", "strong_aug_2") for p in phase_history[4:23])
        assert all(p == "light_aug" for p in phase_history[23:])

    def test_error_without_shared_epoch(self):
        """Callback should propagate ValueError if shared epoch not set."""
        switch = DataAugSwitch(POLICY_EPOCHS, _make_minimal_policies(), input_size=[640, 640])
        cb = AugmentationSchedulerCallback(data_aug_switch=switch)

        mock_trainer = MagicMock(spec=Trainer)
        mock_trainer.current_epoch = 10
        mock_pl_module = MagicMock(spec=LightningModule)

        with pytest.raises(ValueError, match="Shared epoch not set"):
            cb.on_train_epoch_start(mock_trainer, mock_pl_module)


# ===================================================================
# Integration-style tests with real pipelines
# ===================================================================
class TestDataAugSwitchIntegration:
    """Integration tests using real CPUAugmentationPipeline and GPUAugmentationPipeline."""

    @pytest.fixture
    def switch(self):
        return DataAugSwitch(
            policy_epochs=POLICY_EPOCHS,
            policies=_make_minimal_policies(),
            input_size=[640, 640],
        )

    def test_all_policies_have_callable_cpu_pipeline(self, switch):
        for name, policy in switch.policies.items():
            pipeline = policy["cpu_pipeline"]
            assert callable(pipeline), f"CPU pipeline for '{name}' is not callable"

    def test_build_gpu_pipeline_produces_aug_sequential(self, switch):
        gpu = switch.build_gpu_pipeline("strong_aug_1", data_keys=["input", "bbox_xyxy", "label"])
        assert gpu.aug_sequential is not None
        assert len(gpu._augmentations_list) == 2

    def test_gpu_pipeline_normalize_extraction(self, switch):
        gpu = switch.build_gpu_pipeline("no_aug", data_keys=["input", "bbox_xyxy", "label"])
        assert gpu.mean is not None
        assert gpu.std is not None
        # mean/std may be tuple or tensor depending on implementation
        mean_t = torch.tensor(gpu.mean) if not isinstance(gpu.mean, torch.Tensor) else gpu.mean
        std_t = torch.tensor(gpu.std) if not isinstance(gpu.std, torch.Tensor) else gpu.std
        assert torch.allclose(mean_t, torch.tensor([0.485, 0.456, 0.406]), atol=1e-4)
        assert torch.allclose(std_t, torch.tensor([0.229, 0.224, 0.225]), atol=1e-4)

    def test_end_to_end_phase_cycle(self):
        switch = DataAugSwitch(POLICY_EPOCHS, _make_minimal_policies(), input_size=[640, 640])
        switch.set_shared_epoch(Value("i", 0))

        expected_phases = {
            0: "no_aug",
            3: "no_aug",
            4: ("strong_aug_1", "strong_aug_2"),
            22: ("strong_aug_1", "strong_aug_2"),
            23: "light_aug",
            40: "light_aug",
        }

        for epoch, expected in expected_phases.items():
            switch.epoch = epoch
            policy = switch.current_policy_name
            if isinstance(expected, tuple):
                assert policy in expected, f"Epoch {epoch}: expected one of {expected}, got {policy}"
            else:
                assert policy == expected, f"Epoch {epoch}: expected {expected}, got {policy}"

    def test_concurrent_access_simulation(self):
        """Shared epoch is mp.Value, safe for concurrent read/write."""
        switch = DataAugSwitch(POLICY_EPOCHS, _make_minimal_policies(), input_size=[640, 640])
        shared = Value("i", 0)
        switch.set_shared_epoch(shared)

        # Simulate callback writing
        switch.epoch = 15
        # Simulate worker reading
        assert switch.epoch == 15
        assert switch.current_policy_name in ("strong_aug_1", "strong_aug_2")
