import argparse
from unittest.mock import patch

from otx.cli.tools import find as target_package
from tests.test_suite.e2e_test_system import e2e_pytest_unit


def create_mock_args(task=None, template=None, backbone=None):
    mock_args = argparse.Namespace()
    mock_args.task = task
    mock_args.template = template
    mock_args.backbone = backbone
    return mock_args


@e2e_pytest_unit
def test_generate_backbone_rows():
    backbone_meta = {
        "mock_backbone": {
            "required": ["first", "second"],
            "options": {"first": ["option1", "option2"], "second": ["option1", "option2"]},
            "available": True,
        }
    }
    rows = target_package.generate_backbone_rows(1, "mock_backbone", backbone_meta["mock_backbone"])

    assert rows == [
        ["1", "mock_backbone", "first", "option1, option2"],
        ["", "", "second", "option1, option2"],
    ]


@e2e_pytest_unit
def test_main():
    mock_args = create_mock_args()
    with patch("argparse.ArgumentParser.parse_args", return_value=mock_args):
        result = target_package.main()

    assert result == {"retcode": 0, "task_type": None}


@e2e_pytest_unit
def test_main_with_template():
    mock_args = create_mock_args(template=True)
    with patch("argparse.ArgumentParser.parse_args", return_value=mock_args):
        result = target_package.main()

    assert result == {"retcode": 0, "task_type": None}


@e2e_pytest_unit
def test_main_with_task():
    mock_args = create_mock_args(task="CLASSIFICATION")
    with patch("argparse.ArgumentParser.parse_args", return_value=mock_args):
        with patch.object(target_package.Registry, "filter") as mock_filter:
            with patch("prettytable.PrettyTable.add_row"):
                result = target_package.main()

    assert result == {"retcode": 0, "task_type": "CLASSIFICATION"}
    mock_filter.assert_called_once_with(task_type="CLASSIFICATION")


@e2e_pytest_unit
def test_main_with_backbone():
    mock_args = create_mock_args(backbone=["backbone1"])
    with patch("argparse.ArgumentParser.parse_args", return_value=mock_args):
        with patch.object(target_package.Registry, "get_backbones") as mock_get_backbones:
            with patch("prettytable.PrettyTable.add_rows"):
                result = target_package.main()

    assert result == {"retcode": 0, "task_type": None}
    mock_get_backbones.assert_called_once_with(["backbone1"])
