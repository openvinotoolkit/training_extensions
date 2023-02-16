import json
from pathlib import Path
from typing import Any, Dict, Union

TEST_TYPES = ["train", "export", "deploy", "nncf", "pot"]
TASK_TYPES = [
    "classification",
    "detection",
    "semantic_segmentation",
    "instance_segmentation",
    "action_classification",
    "action_detection",
    "anomaly",
]
TRAIN_TYPES = ["supervised", "semi_supervised", "self_supervised", "class_incr"]
LABEL_TYPES = ["multi_class", "multi_label", "h_label", "supcon"]


def get_result_dict(task_type: str) -> Dict[str, Any]:
    result_dict = {}
    for test_type in TEST_TYPES:
        result_dict[test_type] = {task_type: {}}
        for label_type in LABEL_TYPES:
            result_dict[test_type][task_type][label_type] = {}
            for train_type in TRAIN_TYPES:
                result_dict[test_type][task_type][label_type][train_type] = {}
    return result_dict


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
        task_type (str): ["classification", "detection", "instance segmentation", "semantic segmentation",
                            "action_classification", "action_detection", "anomaly"]
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
        result["regression_criteria"] = reg_config["regression_criteria"][task_type][train_type][label_type]
        result["data_path"] = reg_config["data_path"][task_type][train_type][label_type]
    else:
        result["regression_criteria"] = reg_config["regression_criteria"][task_type]
        result["data_path"] = reg_config["data_path"][task_type]

    return result
