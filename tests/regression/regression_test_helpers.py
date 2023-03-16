# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import json
from pathlib import Path
from typing import Any, Dict, List, Union

from otx.api.entities.model_template import ModelTemplate

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
TRAIN_TYPES = ["supervised", "semi_supervised", "self_supervised", "class_incr", "tiling"]
LABEL_TYPES = ["multi_class", "multi_label", "h_label", "supcon"]

REGRESSION_TEST_EPOCHS = "10"

ANOMALY_DATASET_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


TIME_LOG = {
    "train_time": "Train + val time (sec.)",
    "infer_time": "Infer time (sec.)",
    "export_time": "Export time (sec.)",
    "export_eval_time": "Export eval time (sec.)",
    "deploy_time": "Deploy time (sec.)",
    "deploy_eval_time": "Deploy eval time (sec.)",
    "nncf_time": "NNCF time (sec.)",
    "nncf_eval_time": "NNCF eval time (sec.)",
    "pot_time": "POT time (sec.)",
    "pot_eval_time": "POT eval time (sec.)",
}


def get_result_dict(task_type: str) -> Dict[str, Any]:
    result_dict = {task_type: {}}
    if "anomaly" not in task_type:
        for label_type in LABEL_TYPES:
            result_dict[task_type][label_type] = {}
            for train_type in TRAIN_TYPES:
                result_dict[task_type][label_type][train_type] = {}
                for test_type in TEST_TYPES:
                    result_dict[task_type][label_type][train_type][test_type] = []
    else:
        for test_type in TEST_TYPES:
            result_dict[task_type][test_type] = {}
            for category in ANOMALY_DATASET_CATEGORIES:
                result_dict[task_type][test_type][category] = []

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
    otx_dir: str, task_type: str, train_type: str = "", label_type: str = ""
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

    if "anomaly" not in task_type:
        if train_type == "" or label_type == "":
            raise ValueError()
        result["regression_criteria"] = reg_config["regression_criteria"][task_type][train_type][label_type]
        result["kpi_e2e_train_time_criteria"] = reg_config["kpi_e2e_train_time_criteria"][task_type][train_type][
            label_type
        ]
        result["kpi_e2e_eval_time_criteria"] = reg_config["kpi_e2e_eval_time_criteria"][task_type][train_type][
            label_type
        ]
        result["data_path"] = reg_config["data_path"][task_type][train_type][label_type]
    else:
        result["regression_criteria"] = reg_config["regression_criteria"][task_type]
        result["kpi_e2e_train_time_criteria"] = reg_config["kpi_e2e_train_time_criteria"][task_type]
        result["kpi_e2e_eval_time_criteria"] = reg_config["kpi_e2e_eval_time_criteria"][task_type]
        result["data_path"] = reg_config["data_path"][task_type]

    return result


def get_template_performance(results: List[Dict], template: ModelTemplate):
    """Get proper template performance inside of performance list."""
    performance = None
    for result in results:
        template_name = list(result.keys())[0]
        if template_name == template.name:
            performance = result
            break
    if performance is None:
        raise ValueError("Performance is None.")
    return performance
