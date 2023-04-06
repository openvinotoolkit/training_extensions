from copy import deepcopy

import pytest
import torch

from mmcls.models.classifiers.image import build_backbone
from otx.algorithms.classification.adapters.mmcls.models.classifiers.sam_classifier import (
    ImageClassifier,
    SAMImageClassifier,
)
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


class TestSAMImageClassifier:
    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        mocker.patch.object(ImageClassifier, "__init__", return_value=None)
        SAMImageClassifier._register_state_dict_hook = mocker.MagicMock()
        SAMImageClassifier._register_load_state_dict_pre_hook = mocker.MagicMock()
        self.classifier = SAMImageClassifier()

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
    def classifier(self, request, mocker) -> SAMImageClassifier:
        head_type, loss_type = request.param
        cfg = {
            "backbone": None,
            "head": {
                "type": head_type,
                "num_classes": 5,
                "in_channels": 10,
                "loss": {
                    "type": loss_type,
                },
            },
            "multilabel": False,
            "hierarchical": False,
        }
        if loss_type == "IBLoss":
            cfg["head"]["loss"]["num_classes"] = 5

        class MockBackbone(torch.nn.Module):
            def forward(self, *args, **kwargs):
                return torch.randn([2, 10])

        mocker.patch("mmcls.models.classifiers.image.build_backbone", return_value=MockBackbone())
        classifier = SAMImageClassifier(track_loss_dynamics=True, **cfg)
        return classifier

    @pytest.fixture()
    def data(self):
        img = torch.rand(2, 3, 224, 224)
        gt_label = torch.randint(0, 5, (2, 1))
        return {
            "img": img,
            "gt_label": gt_label,
            "img_metas": [{"entity_id": f"id{idx}"} for idx in range(2)],
        }

    @torch.no_grad()
    @pytest.mark.parametrize("classifier", TESTCASE, indirect=True, ids=lambda x: "-".join(x))
    def test_train_step(self, classifier: SAMImageClassifier, data):
        outputs = classifier.train_step(data)

        assert "loss_dyns" in outputs
        assert "entity_ids" in outputs
        assert "gt_labels" in outputs
