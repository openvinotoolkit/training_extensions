# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import pytest

from otx.cli.registry import Registry, find_and_parse_model_template
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestRegistry:
    @pytest.fixture
    def mock_templates(self, mocker):
        template1 = mocker.MagicMock(framework="OTX001", task_type="Classification", model_template_id="001")
        template2 = mocker.MagicMock(framework="OTX002", task_type="Segmentation", model_template_id="002")
        template3 = mocker.MagicMock(framework="OTX001", task_type="Detection", model_template_id="003")
        return [template1, template2, template3]

    @e2e_pytest_unit
    def test_init(self, mocker, mock_templates):
        mock_glob = mocker.patch("glob.glob")
        mock_glob.return_value = ["./template.yaml"]
        mock_abspath = mocker.patch("os.path.abspath")
        mock_abspath.return_value = "./template.yaml"
        mock_parse_model_template = mocker.patch("otx.cli.registry.registry.parse_model_template")
        mock_parse_model_template.return_value = mock_templates[0]

        r = Registry(templates_dir=".")
        assert r.templates == [mock_templates[0]]

        r = Registry(templates_dir=".", experimental=True)
        assert r.templates == [mock_templates[0]] * 2

        r = Registry(templates=mock_templates)
        assert r.templates == mock_templates

    @e2e_pytest_unit
    def test_init_empty_raise_RuntimeError(self):
        with pytest.raises(RuntimeError):
            Registry()

    @e2e_pytest_unit
    def test_filter(self, mock_templates):
        """Check TestRegistry.filter function working well.

        <Steps>
            1. Create an instance of Registry with the mock templates
            2. Test filtering by framework
            3. Test filtering by task_type
            4. Test filtering by both framework and task_type
        """
        registry = Registry(templates=mock_templates)

        filtered_registry = registry.filter(framework="OTX001")
        assert len(filtered_registry.templates) == 2
        assert filtered_registry.templates[0].task_type == "Classification"
        assert filtered_registry.templates[1].task_type == "Detection"
        filtered_registry = registry.filter(framework="OTX002")
        assert len(filtered_registry.templates) == 1
        assert filtered_registry.templates[0].task_type == "Segmentation"

        filtered_registry = registry.filter(task_type="Detection")
        assert len(filtered_registry.templates) == 1
        assert filtered_registry.templates[0].task_type == "Detection"

        filtered_registry = registry.filter(framework="OTX002", task_type="Segmentation")
        assert len(filtered_registry.templates) == 1
        assert filtered_registry.templates[0].task_type == "Segmentation"

    @e2e_pytest_unit
    def test_get(self, mock_templates):
        registry = Registry(templates=mock_templates)
        template = registry.get("003")
        assert template.task_type == "Detection"

    @e2e_pytest_unit
    def test_get_unexpected(self, mock_templates):
        registry = Registry(templates=mock_templates)
        with pytest.raises(ValueError):
            registry.get("004")

    @e2e_pytest_unit
    def test_get_backbones(self, mocker, mock_templates):
        mock_get_backbone_list = mocker.patch("otx.cli.registry.registry.get_backbone_list")
        mock_get_backbone_list.return_value = ["resnet50", "resnet101"]
        r = Registry(templates=mock_templates)
        assert r.get_backbones(["torch", "tensorflow"]) == {
            "torch": ["resnet50", "resnet101"],
            "tensorflow": ["resnet50", "resnet101"],
        }


@e2e_pytest_unit
def test_find_and_parse_model_template(mocker):
    mock_template = mocker.MagicMock(framework="test", task_type="test", model_template_id="001")
    mock_parse_model_template = mocker.patch("otx.cli.registry.registry.parse_model_template")
    mock_parse_model_template.return_value = mock_template

    assert find_and_parse_model_template("001") == mock_template


@e2e_pytest_unit
def test_find_and_parse_model_template_exists_true(mocker):
    mock_template = mocker.MagicMock(framework="test", task_type="test", model_template_id="001")
    mock_parse_model_template = mocker.patch("otx.cli.registry.registry.parse_model_template")
    mock_parse_model_template.return_value = mock_template
    mock_exists = mocker.patch("os.path.exists")
    mock_exists.return_value = True
    assert find_and_parse_model_template("001") == mock_template
