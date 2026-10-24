# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Ultralytics classification model wrappers.

Covers ``getitune.backend.ultralytics.models.classification``.
"""

from __future__ import annotations

import pytest

from getitune.backend.ultralytics.models import UltralyticsMultiClassClsModel, UltralyticsMultiLabelClsModel
from getitune.types.export import TaskLevelExportParameters
from getitune.types.label import LabelInfo


def _label_info() -> LabelInfo:
    return LabelInfo(label_names=["cat", "dog"], label_ids=["0", "1"], label_groups=[["cat", "dog"]])


class TestClassificationModels:
    """Tests for the Ultralytics classification model wrappers."""

    @pytest.mark.parametrize(
        ("model_cls", "model_name"),
        [
            (UltralyticsMultiClassClsModel, "yolo26n-cls.yaml"),
            (UltralyticsMultiLabelClsModel, "yolo26n-cls.yaml"),
        ],
    )
    def test_task_is_classify(self, model_cls: type, model_name: str) -> None:
        model = model_cls(model_name=model_name, pretrained=False, label_info=_label_info())
        assert model.task == "classify"

    @pytest.mark.parametrize(
        "model_cls",
        [UltralyticsMultiClassClsModel, UltralyticsMultiLabelClsModel],
    )
    def test_default_preprocessing_is_224_identity(self, model_cls: type) -> None:
        model = model_cls(model_name="yolo26n-cls", pretrained=False, label_info=_label_info())
        params = model.data_input_params
        assert params.input_size == (224, 224)
        assert params.mean == (0.0, 0.0, 0.0)
        assert params.std == (1.0, 1.0, 1.0)

    def test_multiclass_export_parameters(self) -> None:
        model = UltralyticsMultiClassClsModel(model_name="yolo26n-cls.yaml", pretrained=False, label_info=_label_info())
        params = model._export_parameters
        assert isinstance(params, TaskLevelExportParameters)
        assert params.model_type == "Classification"
        assert params.task_type == "classification"
        assert params.confidence_threshold is None
        assert params.iou_threshold is None
        assert params.nms_execute is False
        assert params.label_info == _label_info()

    def test_multilabel_label_info_dispatch_builds_single_group(self) -> None:
        """Multilabel models use the shared base dispatch: one group with all labels.

        This matches ``MultilabelClsDataset.label_info`` and the Lightning
        backend's convention, so ``model.label_info`` stays consistent with
        ``DataModule.label_info`` (checked by ``OVEngine.test()``).
        """
        model = UltralyticsMultiLabelClsModel(model_name="yolo26n-cls.yaml", pretrained=False, label_info=3)
        assert isinstance(model.label_info, LabelInfo)
        assert model.label_info.label_names == ["label_0", "label_1", "label_2"]
        assert model.label_info.label_groups == [["label_0", "label_1", "label_2"]]
        assert model.label_info.label_ids == ["0", "1", "2"]

    def test_multilabel_export_parameters(self) -> None:
        model = UltralyticsMultiLabelClsModel(model_name="yolo26n-cls.yaml", pretrained=False, label_info=3)
        params = model._export_parameters
        assert isinstance(params, TaskLevelExportParameters)
        assert params.model_type == "Classification"
        assert params.task_type == "classification"
        assert params.confidence_threshold == 0.5
        assert params.multilabel is True
        assert params.output_raw_scores is True
        assert params.nms_execute is False
        assert isinstance(params.label_info, LabelInfo)
        assert params.label_info.label_groups == [["label_0", "label_1", "label_2"]]

    def test_multilabel_map_metric_key_aligns_with_detection_and_instance_segmentation(self) -> None:
        """``MultiLabelClsMetricCallable`` reports the multi-label mAP metric under the collection

        key ``"map"`` (see ``getitune.metrics.accuracy._multi_label_cls_metric_callable``), matching
        the lowercase convention used by detection/instance-segmentation's native torchmetrics
        ``MeanAveragePrecision`` and the Lightning backend. The Ultralytics backend must map its
        internal ``metrics/map`` key to the same ``val/map`` name.
        """
        assert UltralyticsMultiLabelClsModel.metric_keys["metrics/map"] == "val/map"
