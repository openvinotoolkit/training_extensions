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

import argparse
import os

from otx.api.configuration.helper import create
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.model import ModelEntity
from otx.api.entities.model_template import TaskType
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.subset import Subset
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.entities.train_parameters import TrainParameters
from otx.api.serialization.label_mapper import label_schema_to_bytes
from otx.api.usecases.adapters.model_adapter import ModelAdapter
from otx.cli.registry import find_and_parse_model_template
from otx.cli.utils.config import configure_dataset, override_parameters
from otx.cli.utils.hpo import run_hpo
from otx.cli.utils.importing import get_impl_class
from otx.cli.utils.multi_gpu import MultiGPUManager
from otx.cli.utils.io import read_binary, read_label_schema, save_model_data
from otx.cli.utils.parser import (
    add_hyper_parameters_sub_parser,
    gen_params_dict_from_args,
)
from otx.core.data import get_dataset_adapter


def parse_args():
    """Parses command line arguments.

    It dynamically generates help for hyper-parameters which are specific to particular model template.
    """

    pre_parser = argparse.ArgumentParser(add_help=False)
    if os.path.exists("./template.yaml"):
        template_path = "./template.yaml"
    else:
        pre_parser.add_argument("template")
        parsed, _ = pre_parser.parse_known_args()
        template_path = parsed.template
    # Load template.yaml file.
    template = find_and_parse_model_template(template_path)
    # Get hyper parameters schema.
    hyper_parameters = template.hyper_parameters.data
    assert hyper_parameters

    parser = argparse.ArgumentParser()
    if not os.path.exists("./template.yaml"):
        parser.add_argument("template")
    parser.add_argument("--data", required=False, default="./data.yaml")
    parsed, _ = parser.parse_known_args()
    required = not os.path.exists(parsed.data)

    parser.add_argument(
        "--train-ann-files",
        required=False,
        help="Comma-separated paths to training annotation files.",
    )
    parser.add_argument(
        "--train-data-roots",
        required=required,
        help="Comma-separated paths to training data folders.",
    )
    parser.add_argument(
        "--val-ann-files",
        required=False,
        help="Comma-separated paths to validation annotation files.",
    )
    parser.add_argument(
        "--val-data-roots",
        required=False,
        help="Comma-separated paths to validation data folders.",
    )
    parser.add_argument(
        "--unlabeled-data-roots",
        required=False,
        help="Comma-separated paths to unlabeled data folders",
    )
    parser.add_argument(
        "--unlabeled-file-list",
        required=False,
        help="Comma-separated paths to unlabeled file list",
    )

    parser.add_argument(
        "--load-weights",
        required=False,
        help="Load model weights from previously saved checkpoint.",
    )
    parser.add_argument(
        "--resume-from",
        required=False,
        help="Resume training from previously saved checkpoint",
    )
    parser.add_argument(
        "--save-model-to",
        required=False,
        help="Location where trained model will be stored.",
    )
    parser.add_argument(
        "--work-dir",
        required=False,
        help="Location where the intermediate output of the training will be stored.",
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

    add_hyper_parameters_sub_parser(parser, hyper_parameters)

    return parser.parse_args(), template, hyper_parameters


def main():  # pylint: disable=too-many-branches
    """Main function that is used for model training."""
    # Dynamically create an argument parser based on override parameters.
    args, template, hyper_parameters = parse_args()
    # Get new values from user's input.
    updated_hyper_parameters = gen_params_dict_from_args(args)
    # Override overridden parameters by user's values.
    override_parameters(updated_hyper_parameters, hyper_parameters)

    hyper_parameters = create(hyper_parameters)

    # Get classes for Task, ConfigurableParameters and Dataset.
    task_class = get_impl_class(template.entrypoints.base)
    data_config = configure_dataset(args)

    data_roots = dict(
        train_subset={
            "ann_file": data_config["data"]["train"]["ann-files"],
            "data_root": data_config["data"]["train"]["data-roots"],
        },
    )
    if data_config["data"]["val"]["data-roots"]:
        data_roots["val_subset"] = {
            "ann_file": data_config["data"]["val"]["ann-files"],
            "data_root": data_config["data"]["val"]["data-roots"],
        }
    if data_config["data"]["unlabeled"]["data-roots"]:
        data_roots["unlabeled_subset"] = {
            "data_root": data_config["data"]["unlabeled"]["data-roots"],
            "file_list": data_config["data"]["unlabeled"]["file-list"],
        }
        is_include_unlabel_data = True

    # Datumaro
    datumaro_adapter = get_dataset_adapter(template.task_type)
    datumaro_dataset = datumaro_adapter.import_dataset(
        train_data_roots=data_roots["train_subset"]["data_root"],
        val_data_roots=data_roots["val_subset"]["data_root"],
        unlabeled_data_roots=data_roots["unlabeled_subset"]["data_root"] if is_include_unlabel_data else None,
    )
    dataset, label_schema = datumaro_adapter.convert_to_otx_format(datumaro_dataset)

    environment = TaskEnvironment(
        model=None,
        hyper_parameters=hyper_parameters,
        label_schema=label_schema,
        model_template=template,
    )

    if args.load_weights or args.resume_from:
        ckpt_path = args.resume_from if args.resume_from else args.load_weights
        model_adapters = {
            "weights.pth": ModelAdapter(read_binary(ckpt_path)),
            "resume": bool(args.resume_from),
        }

        if os.path.exists(os.path.join(os.path.dirname(ckpt_path), "label_schema.json")):
            model_adapters.update(
                {"label_schema.json": ModelAdapter(label_schema_to_bytes(read_label_schema(ckpt_path)))}
            )

        environment.model = ModelEntity(
            train_dataset=dataset,
            configuration=environment.get_model_configuration(),
            model_adapters=model_adapters,
        )

    if args.enable_hpo:
        task = run_hpo(args, environment, dataset, template.task_type)
        if task is None:
            print("cannot run HPO for this task. will train a model without HPO.")
            task = task_class(task_environment=environment, output_path=args.work_dir)
    else:
        task = task_class(task_environment=environment, output_path=args.work_dir)

    if args.gpus:
        multigpu_manager = MultiGPUManager(main, args.gpus, args.rdzv_endpoint, args.base_rank, args.world_size)
        if template.task_type in (TaskType.ACTION_CLASSIFICATION, TaskType.ACTION_DETECTION):
            print("Multi-GPU training for action tasks isn't supported yet. A single GPU will be used for a training.")
        elif (
            multigpu_manager.is_available()
            and not template.task_type.is_anomaly  # anomaly tasks don't use this way for multi-GPU training
        ):
            multigpu_manager.setup_multi_gpu_train(task.project_path, hyper_parameters if args.enable_hpo else None)

    output_model = ModelEntity(dataset, environment.get_model_configuration())

    task.train(dataset, output_model, train_parameters=TrainParameters())

    if "save_model_to" not in args or not args.save_model_to:
        args.save_model_to = "./models"
    save_model_data(output_model, args.save_model_to)

    if data_config["data"]["val"]["data-roots"]:
        validation_dataset = dataset.get_subset(Subset.VALIDATION)
        predicted_validation_dataset = task.infer(
            validation_dataset.with_empty_annotations(),
            InferenceParameters(is_evaluation=False),
        )

        resultset = ResultSetEntity(
            model=output_model,
            ground_truth_dataset=validation_dataset,
            prediction_dataset=predicted_validation_dataset,
        )
        task.evaluate(resultset)
        assert resultset.performance is not None
        print(resultset.performance)

    if args.gpus:
        multigpu_manager.finalize()


if __name__ == "__main__":
    main()
