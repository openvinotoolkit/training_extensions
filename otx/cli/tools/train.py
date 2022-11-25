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

import argparse
import os
import shutil
import os.path as osp
import sys
import signal
from functools import partial
from typing import List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from otx.api.configuration.helper import create
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.model import ModelEntity
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.subset import Subset
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.entities.train_parameters import TrainParameters
from otx.api.serialization.label_mapper import label_schema_to_bytes
from otx.api.usecases.adapters.model_adapter import ModelAdapter
from otx.cli.datasets import get_dataset_class
from otx.cli.registry import find_and_parse_model_template
from otx.cli.utils.config import configure_dataset, override_parameters
from otx.cli.utils.hpo import run_hpo
from otx.cli.utils.importing import get_impl_class
from otx.cli.utils.io import (
    generate_label_schema,
    read_binary,
    read_label_schema,
    save_model_data,
)
from otx.cli.utils.parser import (
    add_hyper_parameters_sub_parser,
    gen_params_dict_from_args,
)

# pylint: disable=too-many-locals


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
    required = not os.path.exists("./data.yaml")

    parser.add_argument(
        "--train-ann-files",
        required=required,
        help="Comma-separated paths to training annotation files.",
    )
    parser.add_argument(
        "--train-data-roots",
        required=required,
        help="Comma-separated paths to training data folders.",
    )
    parser.add_argument(
        "--val-ann-files",
        required=required,
        help="Comma-separated paths to validation annotation files.",
    )
    parser.add_argument(
        "--val-data-roots",
        required=required,
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
        help="Load only weights from previously saved checkpoint",
    )
    parser.add_argument(
        "--save-model-to",
        required=False,
        help="Location where trained model will be stored.",
    )
    parser.add_argument(
        "--save-logs-to",
        required=False,
        help="Location where logs will be stored.",
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
        "--multi-gpu-train",
        action="store_true",
        help="Enable multi gpu training. if number of gpu is more than one, then model is trained in each gpu with splited dataset.",
    )

    add_hyper_parameters_sub_parser(parser, hyper_parameters)

    return parser.parse_args(), template, hyper_parameters


def main():
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
    dataset_class = get_dataset_class(template.task_type)

    data_config = configure_dataset(args)

    data_roots = dict(
        train_subset={
            "ann_file": data_config["data"]["train"]["ann-files"],
            "data_root": data_config["data"]["train"]["data-roots"],
        },
        val_subset={
            "ann_file": data_config["data"]["val"]["ann-files"],
            "data_root": data_config["data"]["val"]["data-roots"],
        },
    )
    if data_config["data"]["unlabeled"]["data-roots"]:
        data_roots["unlabeled_subset"] = {
            "data_root": data_config["data"]["unlabeled"]["data-roots"],
            "file_list": data_config["data"]["unlabeled"]["file-list"],
        }

    dataset = dataset_class(**data_roots)

    environment = TaskEnvironment(
        model=None,
        hyper_parameters=hyper_parameters,
        label_schema=generate_label_schema(dataset, template.task_type),
        model_template=template,
    )

    if args.load_weights:
        model_adapters = {
            "weights.pth": ModelAdapter(read_binary(args.load_weights)),
        }
        if os.path.exists(os.path.join(os.path.dirname(args.load_weights), "label_schema.json")):
            model_adapters.update(
                {"label_schema.json": ModelAdapter(label_schema_to_bytes(read_label_schema(args.load_weights)))}
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
            task = task_class(task_environment=environment)
    else:
        environment.work_dir = osp.dirname(args.save_model_to)
        task = task_class(task_environment=environment)

    multi_gpu_processes = []
    if args.multi_gpu_train and not template.task_type.is_anomaly:
        gpu_ids = get_gpu_ids()
        if len(gpu_ids) > 1:
            if args.enable_hpo:
                multi_gpu_processes = run_multi_gpu_train(gpu_ids, hyper_parameters)
            else:
                multi_gpu_processes = run_multi_gpu_train(gpu_ids)
        else:
            print("Number of avilable gpu is lower than 2. Multi GPU training won't be executed.")

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

    if args.save_logs_to:
        tmp_path = task.project_path
        logs_path = os.path.join(args.save_logs_to, tmp_path.split("/")[-1])
        shutil.copytree(tmp_path, logs_path)
        print(f"Save logs: {logs_path}")

    for p in multi_gpu_processes:
        p.join()

def get_gpu_ids():
    num_available_gpu = torch.cuda.device_count()
    cuda_environ = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_environ is None:
        return list(range(num_available_gpu))
    else:
        gpu_ids = []
        for gpu_idx in cuda_environ.split(','):
            gpu_idx = int(gpu_idx)
            if gpu_idx < num_available_gpu:
                gpu_ids.append(gpu_idx)
        return gpu_ids

def multigpu_initilization(rank: int, gpu_ids: List[int]):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    torch.cuda.set_device(gpu_ids[rank])
    dist.init_process_group(backend='nccl', world_size=len(gpu_ids), rank=rank)
    print(f'dist info world_size = {dist.get_world_size()}, rank = {dist.get_rank()}')

def run_multigpu_child_process(rank: int, gpu_ids: List[int]):
    sys.argv.remove('--multi-gpu-train')
    if "--enable-hpo" in sys.argv:
        sys.argv.remove('--enable-hpo')

    multigpu_initilization(rank, gpu_ids)
    main()

def terminate_signal_handler(signum, frame, processes: List[mp.Process]):
    for process in processes:
        print(f"Kill child process {process.pid}")
        try:
            process.kill()
        except Exception:
            pass

    singal_name = {2: "SIGINT", 15: "SIGTERM"}
    print(f"{singal_name[signum]} is sent. process exited.")

    sys.exit(1)

def run_multi_gpu_train(gpu_ids: List[int], optimized_hyper_parameters=None):
    if optimized_hyper_parameters is not None:
        set_optimized_hp_for_child_process(optimized_hyper_parameters)

    processes= []
    spawned_mp = mp.get_context("spawn")
    for rank in range(1, len(gpu_ids)):
        task_p = spawned_mp.Process(target=run_multigpu_child_process, args=(rank, gpu_ids))
        task_p.start()
        processes.append(task_p)

    signal.signal(signal.SIGINT, partial(terminate_signal_handler, processes=processes))
    signal.signal(signal.SIGTERM, partial(terminate_signal_handler, processes=processes))

    multigpu_initilization(0, gpu_ids)

    return processes

def set_optimized_hp_for_child_process(hyper_parameters):
    if "params" not in sys.argv:
        sys.argv.append('params')

    def set_hyper_parameter_in_argv(key, value):
        if key in sys.argv:
            sys.argv[sys.argv.index(key) + 1] = value
        else:
            sys.argv.extend([key, value])
            
    set_hyper_parameter_in_argv(
        "--learning_parameters.learning_rate",
        str(hyper_parameters.learning_parameters.learning_rate)
    )
    set_hyper_parameter_in_argv(
        "--learning_parameters.batch_size",
        str(hyper_parameters.learning_parameters.batch_size)
    )

if __name__ == "__main__":
    main()
