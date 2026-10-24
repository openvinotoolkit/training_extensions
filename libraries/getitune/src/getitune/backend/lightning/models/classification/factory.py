# mypy: disable_error_code=misc

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for getitune classification factory."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from getitune.backend.lightning.models.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from getitune.metrics.accuracy import MultiClassClsMetricCallable

from .multiclass_models import (
    EfficientNetMulticlassCls,
    MobileNetV3MulticlassCls,
    TimmModelMulticlassCls,
    TVModelMulticlassCls,
    VisionTransformerMulticlassCls,
)
from .multilabel_models import (
    EfficientNetMultilabelCls,
    MobileNetV3MultilabelCls,
    TimmModelMultilabelCls,
    TVModelMultilabelCls,
    VisionTransformerMultilabelCls,
)

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from getitune.backend.lightning.models.base import DataInputParams
    from getitune.backend.lightning.schedulers import LRSchedulerListCallable
    from getitune.metrics import MetricCallable
    from getitune.types.label import LabelInfoTypes


class MobileNetV3:
    """Factory class for MobileNetV3 models."""

    @overload
    def __new__(
        cls,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams | dict | None = None,
        task: Literal["multi_class", "multi_label"] = "multi_class",
        freeze_backbone: bool = False,
        model_name: Literal["mobilenetv3_large", "mobilenetv3_small"] = "mobilenetv3_large",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiClassClsMetricCallable,
        torch_compile: bool = False,
    ) -> MobileNetV3MulticlassCls | MobileNetV3MultilabelCls: ...

    def __new__(
        cls,
        task: Literal["multi_class", "multi_label"] = "multi_class",
        **kwargs,
    ) -> MobileNetV3MulticlassCls | MobileNetV3MultilabelCls:
        """Factory method to create MobileNetV3 models based on the task type.

        Args:
            label_info (LabelInfoTypes): The label information.
            data_input_params (DataInputParams | dict | None, optional): The data input parameters that consists
                of input size, mean and std. Defaults to None.
            freeze_backbone (bool, optional): Whether to freeze the backbone during training. Defaults to False.
                Note: only multiclass classification supports this argument.
            model_name (str, optional): The model name. Defaults to "mobilenetv3_large".
            task (Literal["multi_class", "multi_label"], optional): The task type.
                Can be "multi_class" or "multi_label". Defaults to "multi_class".
            optimizer (OptimizerCallable, optional): The optimizer callable. Defaults to DefaultOptimizerCallable.
            scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): The learning rate scheduler callable.
                Defaults to DefaultSchedulerCallable.
            metric (MetricCallable, optional): The metric callable. Defaults to MultiClassClsMetricCallable.
            torch_compile (bool, optional): Whether to compile the model using TorchScript. Defaults to False.

        Examples:
        >>> # Basic usage
        >>> model = MobileNetV3(
        ...     task="multi_class",
        ...     label_info=10,
        ...     data_input_params={"input_size": (224, 224),
        ...                        "mean": [0.485, 0.456, 0.406],
        ...                        "std": [0.229, 0.224, 0.225]},
        ...     model_name="mobilenetv3_small",
        ... )

        >>> # Multi-label classification
        >>> model = MobileNetV3(
        ...     task="multi_label",
        ...     model_name="mobilenetv3_large",
        ...     data_input_params={"input_size": (224, 224),
        ...                        "mean": [0.485, 0.456, 0.406],
        ...                        "std": [0.229, 0.224, 0.225]},
        ...     label_info=[1, 5, 10]  # Multi-label setup
        ... )
        """
        if task == "multi_class":
            return MobileNetV3MulticlassCls(**kwargs)
        if task == "multi_label":
            return MobileNetV3MultilabelCls(**kwargs)
        msg = f"Unsupported task type: {task}"
        raise ValueError(msg)


class EfficientNet:
    """Factory class for EfficientNet models."""

    @overload
    def __new__(
        cls,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams | dict | None = None,
        task: Literal["multi_class", "multi_label"] = "multi_class",
        model_name: Literal[
            "efficientnet_b0",
            "efficientnet_b1",
            "efficientnet_b2",
            "efficientnet_b3",
            "efficientnet_b4",
            "efficientnet_b5",
            "efficientnet_b6",
            "efficientnet_b7",
            "efficientnet_b8",
        ] = "efficientnet_b0",
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiClassClsMetricCallable,
        torch_compile: bool = False,
    ) -> EfficientNetMulticlassCls | EfficientNetMultilabelCls: ...

    def __new__(
        cls,
        task: Literal["multi_class", "multi_label"] = "multi_class",
        **kwargs,
    ) -> EfficientNetMulticlassCls | EfficientNetMultilabelCls:
        """Factory method to create EfficientNet models based on the task type.

        Args:
            label_info (LabelInfoTypes): The label information.
            data_input_params (DataInputParams | dict | None, optional): The data input parameters that consists
                of input size, mean and std. Defaults to None.
            freeze_backbone (bool, optional): Whether to freeze the backbone during training. Defaults to False.
                Note: only multiclass classification supports this argument.
            model_name (Literal["efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
                                 "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
                                 "efficientnet_b8"], optional): The model name. Defaults to "efficientnet_b0".
            task (Literal["multi_class", "multi_label"], optional): The task type.
                Can be "multi_class" or "multi_label". Defaults to "multi_class".
            optimizer (OptimizerCallable, optional): The optimizer callable. Defaults to DefaultOptimizerCallable.
            scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): The learning rate scheduler callable.
                Defaults to DefaultSchedulerCallable.
            metric (MetricCallable, optional): The metric callable. Defaults to MultiClassClsMetricCallable.
            torch_compile (bool, optional): Whether to compile the model using TorchScript. Defaults to False.

        Examples:
            >>> # Basic usage
            >>> model = EfficientNet(
            ...     task="multi_class",
            ...     label_info=10,
            ...     data_input_params={"input_size": (224, 224),
            ...                        "mean": [123.675, 116.28, 103.53],
            ...                        "std": [58.395, 57.12, 57.375]},
            ...     model_name="efficientnet_b0",
            ... )
        """
        if task == "multi_class":
            return EfficientNetMulticlassCls(**kwargs)
        if task == "multi_label":
            return EfficientNetMultilabelCls(**kwargs)
        msg = f"Unsupported task type: {task}"
        raise ValueError(msg)


class TimmModel:
    """Factory class for TimmModel models."""

    @overload
    def __new__(
        cls,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams | dict | None = None,
        task: Literal["multi_class", "multi_label"] = "multi_class",
        model_name: str = "tf_efficientnetv2_s.in21k",
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiClassClsMetricCallable,
        torch_compile: bool = False,
    ) -> TimmModelMulticlassCls | TimmModelMultilabelCls: ...

    def __new__(
        cls,
        task: Literal["multi_class", "multi_label"] = "multi_class",
        **kwargs,
    ) -> TimmModelMulticlassCls | TimmModelMultilabelCls:
        """Factory method to create Timm models based on the task type.

        This class allows users to create models for multi-class or multi-label
        classification by specifying the `task` parameter.
        Users can select any model available in the Timm library (over 1400+ models as of 2026)
        by providing its name to the `model_name` parameter.
        To explore all available models, use `timm.list_models()` or `TimmModel.list_model()`.

        Note:
        - If you wish to use Vision Transformer (ViT) models, it is recommended to use the `VisionTransformer`
            implementation provided by getitune for better integration and support.

        Args:
            label_info (LabelInfoTypes): The label information.
            data_input_params (DataInputParams | dict | None, optional): The data input parameters that consists
                of input size, mean and std. Defaults to None.
            freeze_backbone (bool, optional): Whether to freeze the backbone during training.
                Note: only multiclass classification supports this argument. Defaults to False.
            model_name (str, optional): The model name. Defaults to "tf_efficientnetv2_s.in21k".
                You can find all available models at timm.list_models() or using TimmModel.list_model().
            task (Literal["multi_class", "multi_label"], optional): The task type.
                Can be "multi_class" or "multi_label". Defaults to "multi_class".
            optimizer (OptimizerCallable): The optimizer callable. Defaults to DefaultOptimizerCallable.
            scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): The learning rate scheduler callable.
                Defaults to DefaultSchedulerCallable.
            metric (MetricCallable, optional): The metric callable. Defaults to MultiClassClsMetricCallable.
            torch_compile (bool, optional): Whether to compile the model using TorchScript. Defaults to False.

        Examples:
            >>> # Basic usage
            >>> model = TimmModel(
            ...     task="multi_class",
            ...     label_info=10,
            ...     data_input_params={"input_size": (224, 224),
            ...                        "mean": [123.675, 116.28, 103.53],
            ...                        "std": [58.395, 57.12, 57.375]},
            ...     model_name="tf_efficientnetv2_s.in21k",
            ... )
            >>> # Multi-label classification with timm optimizer
            >>> from getitune.backend.lightning.models.classification.utils.timm import TimmOptimizer
            >>> model = TimmModel(
            ...     task="multi_label",
            ...     optimizer = TimmOptimizer(lr=0.01, weight_decay=0.001),
            ...     model_name="tf_efficientnetv2_s.in21k",
            ...     data_input_params={"input_size": (224, 224),
            ...                        "mean": [123.675, 116.28, 103.53],
            ...                        "std": [58.395, 57.12, 57.375]},
            ...     label_info=["1", "5", "10"]  # Multi-label setup
            ... )
        """
        if task == "multi_class":
            return TimmModelMulticlassCls(**kwargs)
        if task == "multi_label":
            return TimmModelMultilabelCls(**kwargs)
        msg = f"Unsupported task type: {task}"
        raise ValueError(msg)

    @staticmethod
    def list_models() -> list[str]:
        """List available Timm models."""
        from timm import list_models

        return list_models(pretrained=True)


class TVModel:
    """Factory class for Torch Vision models."""

    @overload
    def __new__(
        cls,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams | dict | None = None,
        task: Literal["multi_class", "multi_label"] = "multi_class",
        model_name: str = "efficientnet_v2_s",
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiClassClsMetricCallable,
        torch_compile: bool = False,
    ) -> TVModelMulticlassCls | TVModelMultilabelCls: ...

    def __new__(
        cls,
        task: Literal["multi_class", "multi_label"] = "multi_class",
        **kwargs,
    ) -> TVModelMulticlassCls | TVModelMultilabelCls:
        """Factory to create TV models based on the task type.

        This class allows users to create models for multi-class, multi-label,
        classification by specifying the `task` parameter.
        You can select any model available in the TorchVision library (over 40 models as of 2025)
        by providing its name to the `model_name` parameter.
        To explore all available models, use `torchvision.models.list_models()` or `TVModel.list_models()`.

        Args:
            label_info (LabelInfoTypes): The label information.
            data_input_params (DataInputParams | dict | None, optional): The data input parameters that consists
                of input size, mean and std. Defaults to None.
            freeze_backbone (bool, optional): Whether to freeze the backbone during training.
                Note: only multiclass classification supports this argument. Defaults to False.
            model_name (str, optional): The model name. Defaults to "efficientnet_v2_s".
            task (Literal["multi_class", "multi_label"], optional): The task type.
                Can be "multi_class" or "multi_label". Defaults to "multi_class".
            optimizer (OptimizerCallable, optional): The optimizer callable. Defaults to DefaultOptimizerCallable.
            scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): The learning rate scheduler callable.
                Defaults to DefaultSchedulerCallable.
            metric (MetricCallable, optional): The metric callable. Defaults to MultiClassClsMetricCallable.
            torch_compile (bool, optional): Whether to compile the model using TorchScript. Defaults to False.

        Examples:
            >>> # Basic usage
            >>> model = TVModel(
            ...     task="multi_class",
            ...     label_info=10,
            ...     data_input_params={"input_size": (224, 224),
            ...                        "mean": [123.675, 116.28, 103.53],
            ...                        "std": [58.395, 57.12, 57.375]},
            ...     model_name="efficientnet_v2_s",
            ... )
            ... # Multi-label classification
            >>> model = TVModel(
            ...     task="multi_label",
            ...     model_name="mobilenet_v3_small",
            ...     data_input_params={"input_size": (224, 224),
            ...                        "mean": [123.675, 116.28, 103.53],
            ...                        "std": [58.395, 57.12, 57.375]},
            ...     label_info=[1, 5, 10]  # Multi-label setup
            ... )
        """
        if task == "multi_class":
            return TVModelMulticlassCls(**kwargs)
        if task == "multi_label":
            return TVModelMultilabelCls(**kwargs)
        msg = f"Unsupported task type: {task}"
        raise ValueError(msg)

    @staticmethod
    def list_models() -> list[str]:
        """List available Torch Vision models."""
        from torchvision.models import list_models

        return list_models()


class VisionTransformer:
    """Factory class for VisionTransformer models."""

    @overload
    def __new__(
        cls,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams | dict | None = None,
        task: Literal["multi_class", "multi_label"] = "multi_class",
        model_name: Literal[
            "vit-tiny",
            "vit-small",
            "vit-base",
            "vit-large",
            "dinov2-small",
            "dinov2-base",
            "dinov2-large",
            "dinov2-giant",
        ] = "vit-tiny",
        freeze_backbone: bool = False,
        lora: bool = False,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiClassClsMetricCallable,
        torch_compile: bool = False,
    ) -> VisionTransformerMulticlassCls | VisionTransformerMultilabelCls: ...

    def __new__(
        cls,
        task: Literal["multi_class", "multi_label"] = "multi_class",
        **kwargs,
    ) -> VisionTransformerMulticlassCls | VisionTransformerMultilabelCls:
        """Factory to create VisionTransformer models based on the task type.

        This class supports multi-class and multi-label classification tasks.
        It provides VIT backbones (tiny to large) and DINOv2 backbones (small to giant).

        Args:
            label_info (LabelInfoTypes): The label information.
            data_input_params (DataInputParams | dict | None, optional): The data input parameters that consists
                of input size, mean and std. Defaults to None.
            freeze_backbone (bool, optional): Whether to freeze the backbone during training.
                Note: only multiclass classification supports this argument. Defaults to False.
            model_name (Literal["vit-tiny", "vit-small", "vit-base", "vit-large",
                                "dinov2-small", "dinov2-base", "dinov2-large", "dinov2-giant"], optional):
                The model name. Defaults to "vit-tiny".
            task (Literal["multi_class", "multi_label"], optional): The task type.
                Can be "multi_class" or "multi_label". Defaults to "multi_class".
            optimizer (OptimizerCallable, optional): The optimizer callable. Defaults to DefaultOptimizerCallable.
            scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): The learning rate scheduler callable.
                Defaults to DefaultSchedulerCallable.
            metric (MetricCallable, optional): The metric callable. Defaults to MultiClassClsMetricCallable.
            torch_compile (bool, optional): Whether to compile the model using TorchScript. Defaults to False.

        Examples:
            >>> # Basic usage
            >>> model = VisionTransformer(
            ...     task="multi_class",
            ...     label_info=10,
            ...     data_input_params={"input_size": (224, 224),
            ...                        "mean": [123.675, 116.28, 103.53],
            ...                        "std": [58.395, 57.12, 57.375]},
            ...     model_name="vit-tiny",
            ... )
            >>> # Multi-label classification
            >>> model = VisionTransformer(
            ...     task="multi_label",
            ...     model_name="vit-small",
            ...     data_input_params={"input_size": (224, 224),
            ...                        "mean": [123.675, 116.28, 103.53],
            ...                        "std": [58.395, 57.12, 57.375]},
            ...     label_info=[1, 5, 10]  # Multi-label setup
            ... )
        """
        if task == "multi_class":
            return VisionTransformerMulticlassCls(**kwargs)
        if task == "multi_label":
            return VisionTransformerMultilabelCls(**kwargs)
        msg = f"Unsupported task type: {task}"
        raise ValueError(msg)
