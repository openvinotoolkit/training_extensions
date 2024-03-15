from pathlib import Path
from pprint import pformat
from otx.algorithms.common.utils.utils import is_xpu_available

from otx.api.entities.model_template import ModelTemplate
from otx.cli.utils.report import (
    data_config_to_str,
    env_info_to_str,
    get_otx_report,
    sub_title_to_str,
    task_config_to_str,
    template_info_to_str,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class MockModelTemplate(ModelTemplate):
    """Mock class for ModelTemplate."""

    def __init__(self):
        self.test1 = "abc"
        self.test2 = "cba"


@e2e_pytest_unit
def test_sub_title_to_str():
    expected = "-" * 60 + "\n\n" + "test" + "\n\n" + "-" * 60 + "\n"
    result = sub_title_to_str("test")
    assert expected == result


@e2e_pytest_unit
def test_env_info_to_str(mocker):
    expected = "\tOTX: 1.2\n"
    mocker.patch("mmcv.utils.env.collect_env", return_value={"OTX": "1.2"})
    result = env_info_to_str()
    if is_xpu_available():
        assert expected in result
    else:
        assert expected == result


@e2e_pytest_unit
def test_template_info_to_str():
    mock_template = MockModelTemplate()
    expected = f"\ttest1: {pformat('abc')}\n" + f"\ttest2: {pformat('cba')}\n"
    result = template_info_to_str(mock_template)
    assert expected == result


@e2e_pytest_unit
def test_data_config_to_str():
    data_config = {
        "train_subset": {"data-roots": "aaa"},
        "val_subset": {"data-roots": "aaa"},
    }
    expected = "train_subset:\n\tdata-roots: aaa\nval_subset:\n\tdata-roots: aaa\n"
    result = data_config_to_str(data_config)
    assert expected == result


@e2e_pytest_unit
def test_task_config_to_str():
    task_config = {"a": "b", "b": "c"}
    expected = "a: 'b'\nb: 'c'\n"
    result = task_config_to_str(task_config)
    assert expected == result


@e2e_pytest_unit
def test_get_otx_report(tmp_dir):
    report_path = Path(tmp_dir) / "report.log"
    model_template = MockModelTemplate()
    task_config = {"a": "b", "b": "c"}
    data_config = {
        "train_subset": {"data-roots": "aaa"},
        "val_subset": {"data-roots": "aaa"},
    }
    get_otx_report(
        model_template=model_template,
        task_config=task_config,
        data_config=data_config,
        results={"time": "0:01"},
        output_path=str(report_path),
    )
    assert report_path.exists()
