"""OTX CLI entry point."""

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

import argparse
import sys
import uuid
import yaml
from pathlib import Path


from .build import main as otx_build
from .demo import main as otx_demo
from .deploy import main as otx_deploy
from .eval import main as otx_eval
from .explain import main as otx_explain
from .export import main as otx_export
from .find import main as otx_find
from .optimize import main as otx_optimize
from .train import main as otx_train

__all__ = [
    "otx_demo",
    "otx_deploy",
    "otx_eval",
    "otx_explain",
    "otx_export",
    "otx_find",
    "otx_train",
    "otx_optimize",
    "otx_build",
]


def parse_args():
    """Parses command line arguments."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("template", nargs="?", default=None)
    parser.add_argument(
        "--workspace",
        help="Location where the intermediate output of the training will be stored.",
        default=None,
    )

    return parser.parse_known_args()


def main():
    """Entry point for OTX CLI.

    This function is a single entry point for all OTX CLI related operations:
      - build
      - demo
      - deploy
      - eval
      - explain
      - export
      - find
      - train
      - optimize
    """

    dataset_path = "/home/eunwoo/work/exp_resource/dataset/diopsis/12"
    args = parse_args()[0]
    template = args.template

    with open(template, "r") as f:
        template_file = yaml.safe_load(f)
    model_name = template_file["name"]

    if args.workspace is None:
        workspace_path = f"./{model_name}_{uuid.uuid4().hex}"
    else:
        workspace_path = args.workspace

    for opt in ["otx_train", "otx_eval", "otx_export", "otx_eval"]:
        argv_list = [
            " ".join(opt.split("_")),
            template,
        ]

        if opt == "otx_train":
            argv_list.extend([
                "--train-data-roots",
                dataset_path,
                "--val-data-roots",
                dataset_path,
                "--workspace",
                workspace_path,
                "--track-resource-usage",
                "params",
                "--learning_parameters.num_iters",
                "1"
            ])
        elif opt == "otx_eval":
            ov_path = list(Path(workspace_path).rglob("openvino.bin"))
            if ov_path:
                ov_path = ov_path[0]
                weight_args = ["--load-weights", str(ov_path)]
                output_path = str(ov_path.parents[1])
            else:
                weight_args = ["--workspace", workspace_path]
                output_path = str(Path(workspace_path) / "outputs" / "latest_trained_model")

            argv_list.extend(weight_args)
            argv_list.extend([
                "--test-data-roots",
                dataset_path,
                "--output",
                output_path
            ])
        elif opt == "otx_export":
            argv_list.extend([
                "--workspace",
                workspace_path,
            ])

        sys.argv = argv_list
        globals()[opt]()
    
    return dict(retcode=0)


if __name__ == "__main__":
    main()
