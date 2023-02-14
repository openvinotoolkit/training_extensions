import json
from pathlib import Path
from typing import Any, Dict, Union

from otx.api.entities.model_template import ModelTemplate
from otx.cli.utils.tests import get_template_dir


def load_regression_config(otx_dir: str) -> Dict[str, Any]:
    """Load regression config from path.

    Args:
        otx_dir (str): The path of otx root directory

    Returns:
        Dict[str, Any]: The dictionary that includes data roots
    """
    root_path = Path(otx_dir)
    with open(root_path / ("tests/regression/regression_config.json"), "r") as f:
        reg_config = json.load(f)
    return reg_config


def load_regression_configuration(
    otx_dir: str, task_type: str, train_type: str, label_type: str
) -> Dict[str, Union[str, int, float]]:
    """Load dataset path according to task, train, label types.

    Args:
        otx_dir (str): The path of otx root directoy
        task_type (str): ["classification", "detection", "segmentation", ...]
        train_type (str): ["supervised", "semi_supervised", "self_supervised", "class_incr"]
        label_type (str): ["multi_class", "multi_label", "h_label", "supcon"]

    Returns:
        Dict[str, Union[int, float]]: The dictionary that includes model criteria
    """
    reg_config = load_regression_config(otx_dir)
    result: Dict[str, Union[str, int, float]] = {
        "data_path": "",
        "model_criteria": 0,
    }

    if task_type != "anomaly":
        result["model_criteria"] = reg_config["model_criteria"][task_type][train_type][label_type]
        result["data_path"] = reg_config["data_path"][task_type][train_type][label_type]
    else:
        result["model_criteria"] = reg_config["model_criteria"][task_type]
        result["data_path"] = reg_config["data_path"][task_type]

    return result


def test_model_performance(
    dir_path: str, template: ModelTemplate, criteria: Union[int, float], threshold: float = 0.05
):
    """Check the model performance.

    Args:
        performance_json_path (str): The path of performance file
        criteria (Union[int, float]): The criteria of model
        threshold (float, optional): The threshold of model performance. Defaults to 0.05.
    """
    template_work_dir = get_template_dir(template, dir_path)
    performance_json_path = f"{template_work_dir}/trained_{template.model_template_id}/performance.json"
    with open(performance_json_path) as read_file:
        trained_performance = json.load(read_file)

    modified_criteria = criteria - (criteria * threshold)
    for k in trained_performance.keys():
        assert (
            trained_performance[k] <= modified_criteria
        ), f"Current model performance: ({trained_performance[k]}) <= criteria: ({modified_criteria})."
