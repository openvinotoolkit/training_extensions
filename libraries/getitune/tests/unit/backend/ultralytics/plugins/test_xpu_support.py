# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Phase 3 — XPU device support."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from getitune.backend.ultralytics.engine import UltralyticsEngine
from getitune.types.device import DeviceType

# _resolve_device() tests


class TestResolveDevice:
    """Tests for ``UltralyticsEngine._resolve_device``."""

    def test_auto_prefers_xpu_over_cuda(self) -> None:
        with (
            patch("getitune.backend.ultralytics.engine.is_xpu_available", return_value=True),
            patch("getitune.backend.ultralytics.engine.torch.cuda.is_available", return_value=True),
        ):
            result = UltralyticsEngine._resolve_device("auto")
            assert result == torch.device("xpu")

    def test_auto_falls_back_to_cuda(self) -> None:
        with (
            patch("getitune.backend.ultralytics.engine.is_xpu_available", return_value=False),
            patch("getitune.backend.ultralytics.engine.torch.cuda.is_available", return_value=True),
        ):
            result = UltralyticsEngine._resolve_device("auto")
            assert result == torch.device("cuda")

    def test_auto_falls_back_to_cpu(self) -> None:
        with (
            patch("getitune.backend.ultralytics.engine.is_xpu_available", return_value=False),
            patch("getitune.backend.ultralytics.engine.torch.cuda.is_available", return_value=False),
        ):
            result = UltralyticsEngine._resolve_device("auto")
            assert result == torch.device("cpu")

    def test_explicit_xpu_string(self) -> None:
        result = UltralyticsEngine._resolve_device("xpu")
        assert result == torch.device("xpu")

    def test_explicit_xpu_with_index(self) -> None:
        result = UltralyticsEngine._resolve_device("xpu:0")
        assert result == torch.device("xpu:0")

    def test_explicit_cuda_string(self) -> None:
        result = UltralyticsEngine._resolve_device("cuda")
        assert result == torch.device("cuda")

    def test_bare_integer_maps_to_cuda(self) -> None:
        result = UltralyticsEngine._resolve_device("0")
        assert result == torch.device("cuda:0")

    def test_bare_integer_index_1(self) -> None:
        result = UltralyticsEngine._resolve_device("1")
        assert result == torch.device("cuda:1")

    def test_cpu_string(self) -> None:
        result = UltralyticsEngine._resolve_device("cpu")
        assert result == torch.device("cpu")

    def test_cuda_with_index(self) -> None:
        result = UltralyticsEngine._resolve_device("cuda:1")
        assert result == torch.device("cuda:1")

    def test_device_type_xpu(self) -> None:
        result = UltralyticsEngine._resolve_device(DeviceType.xpu)
        assert result == torch.device("xpu")

    def test_device_type_gpu(self) -> None:
        result = UltralyticsEngine._resolve_device(DeviceType.gpu)
        assert result == torch.device("cuda")

    def test_device_type_cpu(self) -> None:
        result = UltralyticsEngine._resolve_device(DeviceType.cpu)
        assert result == torch.device("cpu")

    def test_device_type_auto_xpu(self) -> None:
        with (
            patch("getitune.backend.ultralytics.engine.is_xpu_available", return_value=True),
            patch("getitune.backend.ultralytics.engine.torch.cuda.is_available", return_value=True),
        ):
            result = UltralyticsEngine._resolve_device(DeviceType.auto)
            assert result == torch.device("xpu")

    def test_device_type_auto_cuda(self) -> None:
        with (
            patch("getitune.backend.ultralytics.engine.is_xpu_available", return_value=False),
            patch("getitune.backend.ultralytics.engine.torch.cuda.is_available", return_value=True),
        ):
            result = UltralyticsEngine._resolve_device(DeviceType.auto)
            assert result == torch.device("cuda")

    def test_returns_torch_device(self) -> None:
        """All return values must be ``torch.device`` instances."""
        result = UltralyticsEngine._resolve_device("cpu")
        assert isinstance(result, torch.device)

    def test_gpu_string_maps_to_cuda(self) -> None:
        result = UltralyticsEngine._resolve_device("gpu")
        assert result == torch.device("cuda")


class TestXPUAwareTrainerMixin:
    """Tests for XPU-specific overrides in the trainer mixin."""

    def test_setup_train_replaces_grad_scaler_on_xpu(self) -> None:
        """After _setup_train, GradScaler must be disabled and model must be bf16 on XPU."""
        from getitune.backend.ultralytics.plugins.xpu_mixin import XPUAwareTrainerMixin

        class FakeBaseTrainer:
            def __init__(self):
                self.device = torch.device("xpu:0")
                self.args = SimpleNamespace(amp=True)
                self.scaler = torch.amp.GradScaler("cuda", enabled=True)
                self.model = torch.nn.Linear(4, 2)

            def _setup_train(self) -> None:
                pass  # parent does nothing in test

        class TestTrainer(XPUAwareTrainerMixin, FakeBaseTrainer):
            pass

        trainer = TestTrainer()
        trainer._setup_train()

        assert not trainer.scaler.is_enabled()
        assert next(trainer.model.parameters()).dtype == torch.bfloat16, "model must be bf16 after _setup_train on XPU"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required — GradScaler auto-disables without it")
    def test_setup_train_keeps_grad_scaler_on_cuda(self) -> None:
        """On CUDA, the mixin must not touch the GradScaler."""
        from getitune.backend.ultralytics.plugins.xpu_mixin import XPUAwareTrainerMixin

        class FakeBaseTrainer:
            def __init__(self):
                self.device = torch.device("cuda:0")
                self.scaler = torch.amp.GradScaler("cuda", enabled=True)

            def _setup_train(self) -> None:
                pass

        class TestTrainer(XPUAwareTrainerMixin, FakeBaseTrainer):
            pass

        trainer = TestTrainer()
        trainer._setup_train()

        # Scaler should remain as parent set it (enabled=True)
        assert trainer.scaler.is_enabled()

    def test_setup_train_bypasses_check_amp_on_xpu(self) -> None:
        """On XPU, _setup_train must monkey-patch check_amp to return False (disable autocast)."""
        from getitune.backend.ultralytics.plugins.xpu_mixin import XPUAwareTrainerMixin

        check_amp_called_with_device = {}

        class FakeBaseTrainer:
            def __init__(self):
                self.device = torch.device("xpu:0")
                self.args = SimpleNamespace(amp=True)
                self.scaler = torch.amp.GradScaler("cpu", enabled=False)
                self.model = torch.nn.Linear(4, 2)

            def _setup_train(self) -> None:
                import ultralytics.engine.trainer as mod

                check_amp_called_with_device["fn"] = mod.check_amp

        class TestTrainer(XPUAwareTrainerMixin, FakeBaseTrainer):
            pass

        trainer = TestTrainer()
        trainer._setup_train()

        # During _setup_train, check_amp should have been replaced with a
        # lambda that returns False (so Ultralytics' autocast is a no-op).
        patched_fn = check_amp_called_with_device["fn"]
        assert patched_fn("anything") is False, "check_amp should be patched to return False on XPU"

        # After _setup_train, the original check_amp must be restored
        import ultralytics.engine.trainer as mod

        assert mod.check_amp is not patched_fn, "check_amp must be restored after _setup_train"

    def test_get_memory_xpu_returns_float(self) -> None:
        """_get_memory() should return a float on XPU."""
        from getitune.backend.ultralytics.plugins.xpu_mixin import XPUAwareTrainerMixin

        class FakeBaseTrainer:
            def __init__(self):
                self.device = torch.device("xpu:0")

            def _get_memory(self, fraction: bool = False) -> float:
                return 0.0

        class TestTrainer(XPUAwareTrainerMixin, FakeBaseTrainer):
            pass

        trainer = TestTrainer()

        with patch("torch.xpu.memory_reserved", return_value=1024 * 1024 * 1024):
            result = trainer._get_memory()
            assert isinstance(result, float)
            assert pytest.approx(result, abs=0.01) == 1.0  # 1 GiB

    def test_get_memory_fraction_xpu(self) -> None:
        from getitune.backend.ultralytics.plugins.xpu_mixin import XPUAwareTrainerMixin

        class FakeBaseTrainer:
            def __init__(self):
                self.device = torch.device("xpu:0")

            def _get_memory(self, fraction: bool = False) -> float:
                return 0.0

        class TestTrainer(XPUAwareTrainerMixin, FakeBaseTrainer):
            pass

        trainer = TestTrainer()
        props = MagicMock()
        props.total_memory = 16 * 2**30  # 16 GiB

        with (
            patch("torch.xpu.memory_reserved", return_value=4 * 2**30),
            patch("torch.xpu.get_device_properties", return_value=props),
        ):
            result = trainer._get_memory(fraction=True)
            assert pytest.approx(result, abs=0.01) == 0.25

    def test_clear_memory_xpu_calls_empty_cache(self) -> None:
        from getitune.backend.ultralytics.plugins.xpu_mixin import XPUAwareTrainerMixin

        class FakeBaseTrainer:
            def __init__(self):
                self.device = torch.device("xpu:0")

            def _clear_memory(self, threshold: float | None = None) -> None:
                pass

        class TestTrainer(XPUAwareTrainerMixin, FakeBaseTrainer):
            pass

        trainer = TestTrainer()

        with patch("torch.xpu.empty_cache") as mock_clear:
            trainer._clear_memory()
            mock_clear.assert_called_once()

    def test_clear_memory_cuda_delegates_to_super(self) -> None:
        from getitune.backend.ultralytics.plugins.xpu_mixin import XPUAwareTrainerMixin

        class FakeBaseTrainer:
            def __init__(self):
                self.device = torch.device("cuda:0")
                self.super_called = False

            def _clear_memory(self, threshold: float | None = None) -> None:
                self.super_called = True

        class TestTrainer(XPUAwareTrainerMixin, FakeBaseTrainer):
            pass

        trainer = TestTrainer()
        trainer._clear_memory()
        assert trainer.super_called

    def test_move_batch_to_device_uses_non_blocking_on_xpu(self) -> None:
        """The helper should use non_blocking=True on XPU and cast images to bf16."""
        from getitune.backend.ultralytics.plugins.xpu_mixin import XPUAwareTrainerMixin

        class FakeBaseTrainer:
            def __init__(self):
                self.device = torch.device("xpu:0")
                self.args = SimpleNamespace(amp=True)

        class TestTrainer(XPUAwareTrainerMixin, FakeBaseTrainer):
            pass

        trainer = TestTrainer()

        img_tensor = MagicMock(spec=torch.Tensor)
        moved_tensor = MagicMock(spec=torch.Tensor)
        bf16_tensor = MagicMock(spec=torch.Tensor)
        img_tensor.to.return_value = moved_tensor
        moved_tensor.bfloat16.return_value = bf16_tensor

        label_tensor = MagicMock(spec=torch.Tensor)
        label_tensor.to.return_value = label_tensor

        batch = {"img": img_tensor, "cls": label_tensor, "non_tensor": "value"}

        with patch(
            "getitune.backend.ultralytics.plugins.xpu_mixin.isinstance",
            side_effect=lambda o, t: (
                True if t is torch.Tensor and o in (img_tensor, label_tensor) else isinstance(o, t)
            ),
        ):
            result = trainer._move_batch_to_device(batch)

        img_tensor.to.assert_called_once_with(torch.device("xpu:0"), non_blocking=True)
        moved_tensor.bfloat16.assert_called_once()
        assert result["img"] is bf16_tensor, "images must be cast to bf16 on XPU"
        assert result["non_tensor"] == "value"

    def test_move_batch_to_device_no_bf16_on_cuda(self) -> None:
        """On CUDA, _move_batch_to_device should NOT cast to bf16."""
        from getitune.backend.ultralytics.plugins.xpu_mixin import XPUAwareTrainerMixin

        class FakeBaseTrainer:
            def __init__(self):
                self.device = torch.device("cuda:0")

        class TestTrainer(XPUAwareTrainerMixin, FakeBaseTrainer):
            pass

        trainer = TestTrainer()

        tensor = MagicMock(spec=torch.Tensor)
        tensor.to.return_value = tensor
        batch = {"img": tensor}

        with patch(
            "getitune.backend.ultralytics.plugins.xpu_mixin.isinstance",
            side_effect=lambda o, t: True if t is torch.Tensor and o is tensor else isinstance(o, t),
        ):
            _ = trainer._move_batch_to_device(batch)

        tensor.to.assert_called_once_with(torch.device("cuda:0"), non_blocking=True)
        tensor.bfloat16.assert_not_called()

    def test_setup_train_converts_model_to_bf16(self) -> None:
        """On XPU, _setup_train must convert model to bf16."""
        from getitune.backend.ultralytics.plugins.xpu_mixin import XPUAwareTrainerMixin

        class FakeBaseTrainer:
            def __init__(self):
                self.device = torch.device("xpu:0")
                self.args = SimpleNamespace(amp=True)
                self.scaler = torch.amp.GradScaler("cpu", enabled=False)
                self.model = torch.nn.Linear(4, 2)

            def _setup_train(self) -> None:
                pass

        class TestTrainer(XPUAwareTrainerMixin, FakeBaseTrainer):
            pass

        trainer = TestTrainer()
        assert next(trainer.model.parameters()).dtype == torch.float32, "model starts fp32"
        trainer._setup_train()
        assert next(trainer.model.parameters()).dtype == torch.bfloat16, "model must be bf16 after setup"

    def test_setup_train_keeps_ema_fp32(self) -> None:
        """On XPU, _setup_train must keep ModelEMA in fp32 for precise averaging."""
        from getitune.backend.ultralytics.plugins.xpu_mixin import XPUAwareTrainerMixin

        ema_model = torch.nn.Linear(4, 2)

        class FakeBaseTrainer:
            def __init__(self):
                self.device = torch.device("xpu:0")
                self.args = SimpleNamespace(amp=True)
                self.scaler = torch.amp.GradScaler("cpu", enabled=False)
                self.model = torch.nn.Linear(4, 2)
                self.ema = MagicMock()
                self.ema.ema = ema_model

            def _setup_train(self) -> None:
                pass

        class TestTrainer(XPUAwareTrainerMixin, FakeBaseTrainer):
            pass

        trainer = TestTrainer()
        trainer._setup_train()

        assert next(trainer.model.parameters()).dtype == torch.bfloat16, "model must be bf16"
        assert next(trainer.ema.ema.parameters()).dtype == torch.float32, "EMA must stay fp32"

    def test_validate_does_not_convert_model_when_no_ema(self) -> None:
        """On XPU without EMA, validate() should leave the model dtype alone."""
        from getitune.backend.ultralytics.plugins.xpu_mixin import XPUAwareTrainerMixin

        model_dtype_during_validation = {}

        class FakeBaseTrainer:
            def __init__(self):
                self.device = torch.device("xpu:0")
                self.args = SimpleNamespace(amp=True)
                self.model = torch.nn.Linear(4, 2).bfloat16()

            def validate(self) -> dict:
                model_dtype_during_validation["dtype"] = next(self.model.parameters()).dtype
                return {}

        class TestTrainer(XPUAwareTrainerMixin, FakeBaseTrainer):  # pyrefly: ignore[inconsistent-inheritance]
            pass

        trainer = TestTrainer()
        trainer.validate()

        assert model_dtype_during_validation["dtype"] == torch.bfloat16, "model must stay bf16 during validation"
        assert next(trainer.model.parameters()).dtype == torch.bfloat16, "model must remain bf16 after validation"

    def test_validate_promotes_bf16_loss_to_fp32(self) -> None:
        """On XPU, validate() must expose fp32 loss to Ultralytics metric code."""
        from getitune.backend.ultralytics.plugins.xpu_mixin import XPUAwareTrainerMixin

        loss_dtype_during_validation = {}

        class FakeBaseTrainer:
            def __init__(self):
                self.device = torch.device("xpu:0")
                self.args = SimpleNamespace(amp=True)
                self.loss = torch.tensor(1.5, dtype=torch.bfloat16)
                self.model = torch.nn.Linear(4, 2).bfloat16()

            def validate(self) -> dict:
                loss_dtype_during_validation["dtype"] = self.loss.dtype
                return {"fitness": 0.7}

        class TestTrainer(XPUAwareTrainerMixin, FakeBaseTrainer):  # pyrefly: ignore[inconsistent-inheritance]
            pass

        trainer = TestTrainer()
        trainer.validate()

        assert loss_dtype_during_validation["dtype"] == torch.float32, "Ultralytics must see fp32 loss"
        assert trainer.loss.dtype == torch.float32, "loss stays fp32 after validate"

    def test_validate_skips_conversion_when_ema_exists(self) -> None:
        """On XPU with EMA, validate() should NOT convert model (EMA is already fp32)."""
        from getitune.backend.ultralytics.plugins.xpu_mixin import XPUAwareTrainerMixin

        class FakeBaseTrainer:
            def __init__(self):
                self.device = torch.device("xpu:0")
                self.args = SimpleNamespace(amp=True)
                self.model = torch.nn.Linear(4, 2).bfloat16()
                self.ema = MagicMock()
                self.ema.ema = torch.nn.Linear(4, 2)  # fp32 EMA
                self.validate_called = False

            def validate(self) -> dict:
                self.validate_called = True
                return {}

        class TestTrainer(XPUAwareTrainerMixin, FakeBaseTrainer):  # pyrefly: ignore[inconsistent-inheritance]
            pass

        trainer = TestTrainer()
        trainer.validate()

        assert trainer.validate_called
        assert next(trainer.model.parameters()).dtype == torch.bfloat16, "model must stay bf16 when EMA is used"

    def test_validate_delegates_on_cuda(self) -> None:
        """On CUDA, validate() should NOT touch model dtype."""
        from getitune.backend.ultralytics.plugins.xpu_mixin import XPUAwareTrainerMixin

        class FakeBaseTrainer:
            def __init__(self):
                self.device = torch.device("cuda:0")
                self.validate_called = False

            def validate(self) -> dict:
                self.validate_called = True
                return {}

        class TestTrainer(XPUAwareTrainerMixin, FakeBaseTrainer):  # pyrefly: ignore[inconsistent-inheritance]
            pass

        trainer = TestTrainer()
        trainer.validate()
        assert trainer.validate_called

    def test_setup_train_fp32_on_xpu(self) -> None:
        """On XPU with amp=False, model must stay fp32 and scaler must be disabled."""
        from getitune.backend.ultralytics.plugins.xpu_mixin import XPUAwareTrainerMixin

        bf16_called = []

        class FakeModel(torch.nn.Linear):
            def bfloat16(self) -> FakeModel:
                bf16_called.append(True)
                return super().bfloat16()  # type: ignore[return-value]

        class FakeBaseTrainer:
            def __init__(self):
                self.device = torch.device("xpu:0")
                self.args = SimpleNamespace(amp=False)
                self.scaler = torch.amp.GradScaler("cpu", enabled=False)
                self.model = FakeModel(4, 2)

            def _setup_train(self) -> None:
                pass

        class TestTrainer(XPUAwareTrainerMixin, FakeBaseTrainer):
            pass

        trainer = TestTrainer()
        assert next(trainer.model.parameters()).dtype == torch.float32, "model starts fp32"
        trainer._setup_train()
        assert not trainer.scaler.is_enabled(), "GradScaler must be disabled on XPU"
        assert next(trainer.model.parameters()).dtype == torch.float32, "model must stay fp32 when amp=False"
        assert not bf16_called, "model.bfloat16() must NOT be called when amp=False"

    def test_move_batch_to_device_no_bf16_when_amp_false_on_xpu(self) -> None:
        """On XPU with amp=False, images must NOT be cast to bfloat16."""
        from getitune.backend.ultralytics.plugins.xpu_mixin import XPUAwareTrainerMixin

        class FakeBaseTrainer:
            def __init__(self):
                self.device = torch.device("xpu:0")
                self.args = SimpleNamespace(amp=False)

        class TestTrainer(XPUAwareTrainerMixin, FakeBaseTrainer):
            pass

        trainer = TestTrainer()

        tensor = MagicMock(spec=torch.Tensor)
        tensor.to.return_value = tensor
        batch = {"img": tensor}

        with patch(
            "getitune.backend.ultralytics.plugins.xpu_mixin.isinstance",
            side_effect=lambda o, t: True if t is torch.Tensor and o is tensor else isinstance(o, t),
        ):
            _ = trainer._move_batch_to_device(batch)

        tensor.to.assert_called_once_with(torch.device("xpu:0"), non_blocking=True)
        tensor.bfloat16.assert_not_called()

    def test_validate_skips_conversion_when_amp_false(self) -> None:
        """On XPU with amp=False, validate() must not convert model dtype at all."""
        from getitune.backend.ultralytics.plugins.xpu_mixin import XPUAwareTrainerMixin

        conversion_log = []

        class FakeModel(torch.nn.Linear):
            def float(self) -> FakeModel:
                conversion_log.append("float")
                return super().float()  # type: ignore[return-value]

            def bfloat16(self) -> FakeModel:
                conversion_log.append("bfloat16")
                return super().bfloat16()  # type: ignore[return-value]

        class FakeBaseTrainer:
            def __init__(self):
                self.device = torch.device("xpu:0")
                self.args = SimpleNamespace(amp=False)
                self.model = FakeModel(4, 2)  # fp32 model

            def validate(self) -> dict:
                return {}

        class TestTrainer(XPUAwareTrainerMixin, FakeBaseTrainer):  # pyrefly: ignore[inconsistent-inheritance]
            pass

        trainer = TestTrainer()
        trainer.validate()

        assert not conversion_log, f"No dtype conversions expected when amp=False, got: {conversion_log}"
        assert next(trainer.model.parameters()).dtype == torch.float32, "model must stay fp32"
