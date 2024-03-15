""" Tests for CustomImageClassifier."""
# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os.path as osp
from copy import deepcopy
from typing import Any, Dict

import pytest
import torch
import datumaro as dm

from otx.algorithms.classification.adapters.mmcls.models.classifiers.custom_image_classifier import (
    ImageClassifier,
    CustomImageClassifier,
)
from otx.api.entities.datasets import DatasetEntity
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class OTXMobileNetV3:
    pass


class OTXEfficientNet:
    pass


class OTXEfficientNetV2:
    pass


class MockModule:
    def __init__(self, name):
        if name == "mobilenet":
            self.backbone = OTXMobileNetV3()
            self._state_dict = {
                "classifier.4.weight": torch.rand(1, 1),
                "classifier.4.bias": torch.rand(1),
                "act.weight": torch.rand(1),
                "someweights": torch.rand(2),
            }
        elif name == "effnetv2":
            self.backbone = OTXEfficientNetV2()
            self._state_dict = {
                "model.classifier.weight": torch.rand(1, 1),
                "model.weight": torch.rand(1),
            }
        elif name == "effnet":
            self.backbone = OTXEfficientNet()
            self._state_dict = {
                "features.weight": torch.rand(1, 1),
                "features.active.weight": torch.rand(1, 1),
                "output.weight": torch.rand(1),
                "output.asl.weight": torch.rand(1),
            }
        self.multilabel = False
        self.hierarchical = False
        self.is_export = False

    def state_dict(self):
        return self._state_dict


class TestCustomImageClassifier:
    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        mocker.patch.object(ImageClassifier, "__init__", return_value=None)
        CustomImageClassifier._register_state_dict_hook = mocker.MagicMock()
        CustomImageClassifier._register_load_state_dict_pre_hook = mocker.MagicMock()
        self.classifier = CustomImageClassifier()

    @e2e_pytest_unit
    def test_forward_train(self, mocker):
        img = torch.rand(1, 3, 224, 224)
        gt_labels = torch.rand(1, 1)
        self.classifier.extract_feat = mocker.MagicMock()
        self.classifier.head = mocker.MagicMock()
        self.classifier.augments = None
        self.classifier.multilabel = False
        self.classifier.hierarchical = False
        losses = self.classifier.forward_train(img, gt_labels)

        assert losses is not None

    @pytest.mark.parametrize("name", ["mobilenet", "effnetv2", "effnet"])
    @e2e_pytest_unit
    def test_load_state_dict_pre_hook(self, name):
        self.module = MockModule(name)
        state_dict = self.module.state_dict()
        self.classifier.load_state_dict_pre_hook(self.module, state_dict, prefix="")

        for key in state_dict:
            if name == "mobilenet":
                if "classifier" in key:
                    assert ".3." in key
                if "someweights" in key:
                    assert "backbone." in key
                if "act" in key:
                    assert "head." in key
            elif name == "effnetv2":
                assert "classifier" not in key
                if "model" in key:
                    assert "backbone" in key
            else:
                if "features" in key and "active" not in key:
                    assert "backbone" in key
                elif "active" in key:
                    assert key == "features.active.weight"
                else:
                    assert "head" in key or "fc" in key

    @pytest.mark.parametrize("name", ["mobilenet", "effnetv2", "effnet"])
    @e2e_pytest_unit
    def test_state_dict_hook(self, name):
        self.module = MockModule(name)
        state_dict = self.module.state_dict()
        state_dict_copy = deepcopy(state_dict)
        self.classifier.load_state_dict_pre_hook(self.module, state_dict, prefix="")
        # backward state dict
        self.classifier.state_dict_hook(self.module, state_dict, prefix="")

        assert state_dict.keys() == state_dict_copy.keys()

    @pytest.mark.parametrize("name", ["mobilenet", "effnetv2", "effnet"])
    def test_load_state_dict_mixing_hook(self, name):
        self.module = MockModule(name)
        state_dict = self.module.state_dict()
        chkpt_dict = deepcopy(state_dict)
        model_classes = [0, 1, 2]
        chkpt_classes = [0, 1]
        self.classifier.load_state_dict_mixing_hook(self.module, model_classes, chkpt_classes, chkpt_dict, prefix="")

        assert chkpt_dict.keys() == state_dict.keys()


class TestLossDynamicsTrackingMixin:
    TESTCASE = [
        ("CustomLinearClsHead", "CrossEntropyLoss"),
        ("CustomNonLinearClsHead", "CrossEntropyLoss"),
        ("CustomLinearClsHead", "IBLoss"),
        ("CustomNonLinearClsHead", "IBLoss"),
    ]

    @pytest.fixture()
    def classifier(self, request, mocker, fxt_multi_class_cls_dataset_entity: DatasetEntity) -> CustomImageClassifier:
        head_type, loss_type = request.param
        n_data = len(fxt_multi_class_cls_dataset_entity)
        labels = fxt_multi_class_cls_dataset_entity.get_labels()
        num_classes = len(labels)
        cfg = {
            "backbone": None,
            "head": {
                "type": head_type,
                "num_classes": num_classes,
                "in_channels": 10,
                "loss": {
                    "type": loss_type,
                },
            },
            "multilabel": False,
            "hierarchical": False,
        }
        if loss_type == "IBLoss":
            cfg["head"]["loss"]["num_classes"] = num_classes
            cfg["head"]["loss"]["start"] = 0  # Should be zero

        class MockBackbone(torch.nn.Module):
            def forward(self, *args, **kwargs):
                return torch.randn([n_data, 10])

        mocker.patch("mmcls.models.classifiers.image.build_backbone", return_value=MockBackbone())
        classifier = CustomImageClassifier(track_loss_dynamics=True, **cfg)
        classifier.loss_dyns_tracker.init_with_otx_dataset(fxt_multi_class_cls_dataset_entity)
        return classifier

    @pytest.fixture()
    def data(self, fxt_multi_class_cls_dataset_entity: DatasetEntity):
        n_data = len(fxt_multi_class_cls_dataset_entity)
        labels = fxt_multi_class_cls_dataset_entity.get_labels()
        img = torch.rand(n_data, 3, 8, 8)
        gt_label = torch.arange(0, len(labels), dtype=torch.long).reshape(-1, 1)
        entity_ids = [item.id_ for item in fxt_multi_class_cls_dataset_entity]
        label_ids = [
            label.id_
            for item in fxt_multi_class_cls_dataset_entity
            for ann in item.get_annotations()
            for label in ann.get_labels()
        ]

        assert len(entity_ids) == len(label_ids)

        return {
            "img": img,
            "gt_label": gt_label,
            "img_metas": [
                {"entity_id": entity_id, "label_id": label_id} for entity_id, label_id in zip(entity_ids, label_ids)
            ],
        }

    @torch.no_grad()
    @pytest.mark.parametrize("classifier", TESTCASE, indirect=True, ids=lambda x: "-".join(x))
    def test_train_step(self, classifier: CustomImageClassifier, data: Dict[str, Any], tmp_dir_path: str):
        outputs = classifier.train_step(data)

        assert "loss_dyns" in outputs
        assert "entity_ids" in outputs
        assert "label_ids" in outputs

        n_steps = 3
        for iter in range(n_steps):
            classifier.loss_dyns_tracker.accumulate(outputs, iter)

        export_dir = osp.join(tmp_dir_path, "noisy_label_detection")
        classifier.loss_dyns_tracker.export(export_dir)

        dataset = dm.Dataset.import_from(export_dir, format="datumaro")

        for item in dataset:
            for ann in item.annotations:
                for k, v in ann.attributes.items():
                    assert k in {"iters", "loss_dynamics"}
                    assert len(v) == n_steps
