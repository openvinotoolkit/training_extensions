# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import os
import yaml
from pathlib import Path
from typing import List

from tests.test_suite.run_test_command import check_run


class OTXBenchmark:
    def __init__(
        self,
        datasets: List[str],
        data_root: str = "data",
        num_epoch: int = 0,
        num_repeat: int = 0,
        train_params: dict = {},
        track_resources: bool = False,
        eval_upto: str = "train",
        output_root: str = "otx-benchmark",
        dry_run: bool = False,
        tags: dict = {},
    ):
        self.datasets = datasets
        self.data_root = data_root
        self.num_epoch = num_epoch
        self.num_repeat = num_repeat
        self.train_params = train_params
        self.track_resources = track_resources
        self.eval_upto = eval_upto
        self.output_root = output_root
        self.dry_run = dry_run
        self.tags = tags

    def run(
        self,
        model_id: str,
        train_params: dict = {},
        tags: dict = {},
    ) -> List[str]:
        # Build config file
        cfg = self._build_config(model_id, train_params, tags)
        cfg_dir = Path(self.output_root)
        cfg_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = cfg_dir / "cfg.yaml"
        with open(cfg_path, "w") as cfg_file:
            yaml.dump(cfg, cfg_file, indent=2,)
        cmd = [
            "python",
            "tools/experiment.py",
            "-f",
            cfg_path,
        ]
        if self.dry_run:
            cmd.append("-d")
        # Run benchmark
        check_run(cmd)
        # Load result
        result = None
        return result

    def _build_config(
        self,
        model_id: str,
        train_params: dict = {},
        tags: dict = {},
    ) -> dict:
        all_train_params = self.train_params.copy()
        all_train_params.update(train_params)
        all_tags = self.tags.copy()
        all_tags.update(tags)

        cfg = {}
        cfg["tags"] = all_tags  # metadata
        cfg["output_path"] = os.path.abspath(self.output_root)
        cfg["constants"] = {
            "dataroot": os.path.abspath(self.data_root),
        }
        cfg["variables"] = {
            "model": [model_id],
            "data": self.datasets,
            **{k: [v] for k, v in all_tags.items()},  # To be shown in result file
        }
        cfg["repeat"] = self.num_repeat
        cfg["command"] = []
        resource_param = ""
        if self.track_resources:
            resource_param = "--track-resource-usage all"
        if self.num_epoch > 0:
            all_train_params["learning_parameters.num_iters"] = self.num_epoch
        params_str = " ".join([f"--{k} {v}" for k, v in all_train_params.items()])
        cfg["command"].append(
            "otx train ${model}"
            " --train-data-roots ${dataroot}/${data}"
            " --val-data-roots ${dataroot}/${data}"
            " --deterministic"
            f" {resource_param}"
            f" params {params_str}"
        )
        cfg["command"].append(
            "otx eval"
            " --test-data-roots ${dataroot}/${data}"
        )
        if self.eval_upto == "train":
            return cfg

        cfg["command"].append(
            "otx export"
        )
        cfg["command"].append(
            "otx eval"
            " --test-data-roots ${dataroot}/${data}"
        )
        if self.eval_upto == "export":
            return cfg

        cfg["command"].append(
            "otx optimize"
        )
        cfg["command"].append(
            "otx eval"
            " --test-data-roots ${dataroot}/${data}"
        )
        return cfg
