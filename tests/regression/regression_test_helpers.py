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
import os
from pathlib import Path
from typing import Any, Dict, List, Union

import torch

from otx.api.entities.model_template import ModelTemplate

TEST_TYPES = ["train", "export", "deploy", "nncf", "ptq"]
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
    "ptq_time": "PTQ time (sec.)",
    "ptq_eval_time": "PTQ eval time (sec.)",
}


class RegressionTestConfig(object):
    """Configurations for regression test."""

    def __init__(self, task_type, train_type, label_type, otx_dir, **kwargs):
        self.task_type = task_type
        self.train_type = train_type
        self.label_type = label_type
        self.otx_dir = otx_dir

        self.result_dict = self._init_result_dict()
        result_dir_prefix = kwargs.get("result_dir", "")
        if len(result_dir_prefix) > 0:
            result_dir_prefix = result_dir_prefix + "_"
        self.result_dir = f"/tmp/regression_test_results/{result_dir_prefix}{task_type}"
        Path(self.result_dir).mkdir(parents=True, exist_ok=True)

        self.config_dict = self.load_config()
        self.args = self.config_dict["data_path"]
        train_params = kwargs.get("train_params")
        if train_params is not None:
            self.args["train_params"] = ["params"]
            self.args["train_params"].extend(train_params)

        self.num_cuda_devices = torch.cuda.device_count()
        self.update_gpu_args(self.args, enable_auto_num_worker=kwargs.get("enable_auto_num_worker", True))

    def update_gpu_args(self, args, enable_auto_num_worker=True):
        if self.num_cuda_devices > 1:
            if enable_auto_num_worker:
                if args.get("train_params") is None:
                    args["train_params"] = ["params"]
                train_params = args.get("train_params")
                train_params.append("--learning_parameters.auto_num_workers")
                train_params.append("True")
            args["--gpus"] = "0,1"

    def _load_config_from_json(self) -> Dict[str, Any]:
        """Load regression config from path.

        Returns:
            Dict[str, Any]: The dictionary that includes data roots
        """
        root_path = Path(self.otx_dir)
        with open(root_path / ("tests/regression/regression_config.json"), "r") as f:
            reg_config = json.load(f)
        return reg_config

    def load_config(self, **kwargs) -> Dict[str, Union[int, float]]:
        """load dataset path according to task, train, label types.

        Returns:
            Dict[str, Union[int, float]]: The dictionary that includes model criteria
        """
        task_type = kwargs.get("task_type", self.task_type)
        train_type = kwargs.get("train_type", self.train_type)
        label_type = kwargs.get("label_type", self.label_type)

        reg_config = self._load_config_from_json()
        result: Dict[str, Union[str, int, float]] = {
            "data_path": "",
            "model_criteria": 0,
        }

        data_root = os.environ.get("CI_DATA_ROOT", "/storageserver/pvd_data/otx_data_archive/regression_datasets")

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

            # update data_path using data_root setting
            data_paths = reg_config["data_path"][task_type][train_type][label_type]
            for key, value in data_paths.items():
                data_paths[key] = os.path.join(data_root, value)

            result["data_path"] = data_paths
        else:
            result["regression_criteria"] = reg_config["regression_criteria"][task_type]
            result["kpi_e2e_train_time_criteria"] = reg_config["kpi_e2e_train_time_criteria"][task_type]
            result["kpi_e2e_eval_time_criteria"] = reg_config["kpi_e2e_eval_time_criteria"][task_type]

            # update data_path using data_root setting
            data_paths = reg_config["data_path"][task_type]
            for key, value in data_paths.items():
                if key != "train_params":
                    data_paths[key] = os.path.join(data_root, value)

            result["data_path"] = data_paths

        return result

    def _init_result_dict(self) -> Dict[str, Any]:
        result_dict = {self.task_type: {}}
        if "anomaly" not in self.task_type:
            for label_type in LABEL_TYPES:
                result_dict[self.task_type][label_type] = {}
                for train_type in TRAIN_TYPES:
                    result_dict[self.task_type][label_type][train_type] = {}
                    for test_type in TEST_TYPES:
                        result_dict[self.task_type][label_type][train_type][test_type] = []
        else:
            for test_type in TEST_TYPES:
                result_dict[self.task_type][test_type] = {}
                for category in ANOMALY_DATASET_CATEGORIES:
                    result_dict[self.task_type][test_type][category] = []

        return result_dict

    def get_template_performance(self, template: ModelTemplate, **kwargs):
        """Get proper template performance inside of performance list."""
        performance = None
        results = None
        task_type = kwargs.get("task_type", self.task_type)
        train_type = kwargs.get("train_type", self.train_type)
        label_type = kwargs.get("label_type", self.label_type)

        if "anomaly" in task_type:
            category = kwargs.get("category")
            if category is None:
                raise RuntimeError("missing required keyword arg 'category'")
            results = self.result_dict[task_type]["train"][category]
        else:
            results = self.result_dict[task_type][label_type][train_type]["train"]

        for result in results:
            template_name = list(result.keys())[0]
            if template_name == template.name:
                performance = result
                break
        if performance is None:
            raise ValueError("Performance is None.")
        return performance
