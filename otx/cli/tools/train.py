"""Model training tool."""

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

# pylint: disable=too-many-locals

import datetime
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Optional

from otx.api.entities.model import ModelEntity
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.entities.train_parameters import TrainParameters
from otx.api.serialization.label_mapper import label_schema_to_bytes
from otx.api.usecases.adapters.model_adapter import ModelAdapter
from otx.cli.manager import ConfigManager
from otx.cli.manager.config_manager import TASK_TYPE_TO_SUB_DIR_NAME
from otx.cli.utils.hpo import run_hpo
from otx.cli.utils.importing import get_impl_class
from otx.cli.utils.io import read_binary, read_label_schema, save_model_data
from otx.cli.utils.multi_gpu import MultiGPUManager, is_multigpu_child_process
from otx.cli.utils.parser import (
    MemSizeAction,
    add_hyper_parameters_sub_parser,
    get_parser_and_hprams_data,
)
from otx.cli.utils.report import get_otx_report
from otx.core.data.adapter import get_dataset_adapter


def get_args():
    """Parses command line arguments."""
    parser, hyper_parameters, params = get_parser_and_hprams_data()

    parser.add_argument(
        "--train-data-roots",
        help="Comma-separated paths to training data folders.",
    )
    parser.add_argument("--train-ann-files", help="Comma-separated paths to train annotation files.")
    parser.add_argument(
        "--val-data-roots",
        help="Comma-separated paths to validation data folders.",
    )
    parser.add_argument("--val-ann-files", help="Comma-separated paths to train annotation files.")
    parser.add_argument(
        "--unlabeled-data-roots",
        help="Comma-separated paths to unlabeled data folders",
    )
    parser.add_argument(
        "--unlabeled-file-list",
        help="Comma-separated paths to unlabeled file list",
    )
    parser.add_argument(
        "--train-type",
        help=f"The currently supported options: {TASK_TYPE_TO_SUB_DIR_NAME.keys()}. "
        "Will be difined automatically if no value passed.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--load-weights",
        help="Load model weights from previously saved checkpoint.",
    )
    parser.add_argument(
        "--resume-from",
        help="Resume training from previously saved checkpoint",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Location where outputs (model & logs) will be stored.",
    )
    parser.add_argument(
        "--workspace",
        help="Location where the intermediate output of the training will be stored.",
        default=None,
    )
    parser.add_argument(
        "--enable-hpo",
        action="store_true",
        help="Execute hyper parameters optimization (HPO) before training.",
    )
    parser.add_argument(
        "--hpo-time-ratio",
        default=4,
        type=float,
        help="Expected ratio of total time to run HPO to time taken for full fine-tuning.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        help="Comma-separated indices of GPU. \
              If there are more than one available GPU, then model is trained with multi GPUs.",
    )
    parser.add_argument(
        "--rdzv-endpoint",
        type=str,
        default="localhost:0",
        help="Rendezvous endpoint for multi-node training.",
    )
    parser.add_argument(
        "--base-rank",
        type=int,
        default=0,
        help="Base rank of the current node workers.",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=0,
        help="Total number of workers in a worker group.",
    )
    parser.add_argument(
        "--mem-cache-size",
        action=MemSizeAction,
        dest="params.algo_backend.mem_cache_size",
        type=str,
        required=False,
        help="Size of memory pool for caching decoded data to load data faster. "
        "For example, you can use digits for bytes size (e.g. 1024) or a string with size units "
        "(e.g. 7KiB = 7 * 2^10, 3MB = 3 * 10^6, and 2G = 2 * 2^30).",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Set deterministic to True, default=False.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Set seed for training.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="The data.yaml path want to use in train task.",
    )
    parser.add_argument(
        "--encryption-key",
        type=str,
        default=None,
        help="Encryption key required to train the encrypted dataset. It is not required the non-encrypted dataset",
    )

    sub_parser = add_hyper_parameters_sub_parser(parser, hyper_parameters, return_sub_parser=True)
    # TODO: Temporary solution for cases where there is no template input
    override_param = [f"params.{param[2:].split('=')[0]}" for param in params if param.startswith("--")]
    if not hyper_parameters and "params" in params:
        if "params" in params:
            params = params[params.index("params") :]
            for param in params:
                if param == "--help":
                    print("Without template configuration, hparams information is unknown.")
                elif param.startswith("--"):
                    sub_parser.add_argument(
                        f"{param}",
                        dest=f"params.{param[2:]}",
                    )
    return parser.parse_args(), override_param


def main():
    """Main function that invoke train function with ExitStack."""
    with ExitStack() as exit_stack:
        return train(exit_stack)


def train(exit_stack: Optional[ExitStack] = None):  # pylint: disable=too-many-branches, too-many-statements
    """Function that is used for model training."""
    start_time = time.time()
    mode = "train"
    args, override_param = get_args()

    config_manager = ConfigManager(args, workspace_root=args.workspace, mode=mode)
    # Auto-Configuration for model template
    config_manager.configure_template()

    # Creates a workspace if it doesn't exist.
    if not config_manager.check_workspace():
        config_manager.build_workspace(new_workspace_path=args.workspace)

    # Update Hyper Parameter Configs
    hyper_parameters = config_manager.get_hyparams_config(override_param=override_param)

    # Auto-Configuration for Dataset configuration
    config_manager.configure_data_config(update_data_yaml=config_manager.check_workspace())
    dataset_config = config_manager.get_dataset_config(
        subsets=["train", "val", "unlabeled"],
        hyper_parameters=hyper_parameters,
    )
    dataset_adapter = get_dataset_adapter(**dataset_config)
    dataset, label_schema = dataset_adapter.get_otx_dataset(), dataset_adapter.get_label_schema()
    # Get classes for Task, ConfigurableParameters and Dataset.
    template = config_manager.template
    task_class = get_impl_class(template.entrypoints.base)

    environment = TaskEnvironment(
        model=None,
        hyper_parameters=hyper_parameters,
        label_schema=label_schema,
        model_template=template,
    )

    if args.load_weights or args.resume_from:
        ckpt_path = args.resume_from if args.resume_from else args.load_weights
        model_adapters = {
            "path": ckpt_path,
            "weights.pth": ModelAdapter(read_binary(ckpt_path)),
            "resume": bool(args.resume_from),
        }

        if (Path(ckpt_path).parent / "label_schema.json").exists():
            model_adapters.update(
                {"label_schema.json": ModelAdapter(label_schema_to_bytes(read_label_schema(ckpt_path)))}
            )

        environment.model = ModelEntity(
            train_dataset=dataset,
            configuration=environment.get_model_configuration(),
            model_adapters=model_adapters,  # type: ignore
        )

    if args.enable_hpo:
        environment = run_hpo(
            args.hpo_time_ratio, config_manager.output_path, environment, dataset, config_manager.data_config
        )

    (config_manager.output_path / "logs").mkdir(exist_ok=True, parents=True)

    if args.gpus:
        multigpu_manager = MultiGPUManager(train, args.gpus, args.rdzv_endpoint, args.base_rank, args.world_size)
        if (
            multigpu_manager.is_available()
            and not template.task_type.is_anomaly  # anomaly tasks don't use this way for multi-GPU training
        ):
            multigpu_manager.setup_multi_gpu_train(
                str(config_manager.output_path), hyper_parameters if args.enable_hpo else None
            )
            if exit_stack is not None:
                exit_stack.callback(multigpu_manager.finalize)
            else:
                print(
                    "Warning: due to abstract of ExitStack context, "
                    "if main process raises an error, all processes can be stuck."
                )

    task = task_class(task_environment=environment, output_path=str(config_manager.output_path / "logs"))

    output_model = ModelEntity(dataset, environment.get_model_configuration())

    task.train(
        dataset, output_model, train_parameters=TrainParameters(), seed=args.seed, deterministic=args.deterministic
    )

    model_path = config_manager.output_path / "models"
    save_model_data(output_model, str(model_path))

    end_time = time.time()
    sec = end_time - start_time
    total_time = str(datetime.timedelta(seconds=sec))
    print("otx train time elapsed: ", total_time)
    model_results = {
        "time elapsed": total_time,
        "score": output_model.performance,
        "model_path": str(model_path.absolute()),
    }

    if args.gpus and exit_stack is None:
        multigpu_manager.finalize()
    elif is_multigpu_child_process():
        return

    get_otx_report(
        model_template=config_manager.template,
        task_config=task.config,
        data_config=config_manager.data_config,
        results=model_results,
        output_path=config_manager.output_path / "cli_report.log",
    )
    print(f"otx train CLI report has been generated: {config_manager.output_path / 'cli_report.log'}")

    # Latest model folder symbolic link to models
    latest_path = config_manager.workspace_root / "outputs" / "latest_trained_model"
    if latest_path.exists():
        latest_path.unlink()
    elif not latest_path.parent.exists():
        latest_path.parent.mkdir(exist_ok=True, parents=True)
    latest_path.symlink_to(config_manager.output_path.resolve())

    if not is_multigpu_child_process():
        task.cleanup()

    return dict(retcode=0, template=template.name)


if __name__ == "__main__":
    main()
