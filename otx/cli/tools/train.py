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
import sys
import signal
import threading
from typing import List, Optional
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from otx.api.configuration import ConfigurableParameters
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
        "--gpus",
        type=str,
        help="Comma-separated indcies of GPU. If there are more than one available GPU, then model is trained with multi GPUs.",
    )
    parser.add_argument(
        "--multi-gpu-port",
        default=25000,
        type=int,
        help="port for communication beteween multi GPU processes.",
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
            task = task_class(task_environment=environment, output_path=args.save_logs_to)
    else:
        task = task_class(task_environment=environment, output_path=args.save_logs_to)

    if args.gpus:
        multigpu_manager = MultiGPUManager(args.gpus, str(args.multi_gpu_port))
        if multigpu_manager.is_available(template):
            multigpu_manager.setup_multi_gpu_train(
                task.project_path,
                hyper_parameters if args.enable_hpo else None
            )

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

class MultiGPUManager:
    def __init__(self, gpu_ids: str, multi_gpu_port: str):
        self._gpu_ids = self._get_gpu_ids(gpu_ids)
        self._multi_gpu_port = multi_gpu_port
        self._main_pid = os.getpid()
        self._processes = None

    def _get_gpu_ids(self, gpus: str) -> List[int]:
        num_available_gpu = torch.cuda.device_count()
        gpu_ids = []
        for gpu_id in gpus.split(','):
            if not gpu_id.isnumeric():
                raise RuntimeError("--gpus argument should be numbers separated by ','.")
            gpu_ids.append(int(gpu_id))

        wrong_gpus = []
        for gpu_idx in gpu_ids:
            if gpu_idx >= num_available_gpu:
                wrong_gpus.append(gpu_idx)

        for wrong_gpu in wrong_gpus:
            gpu_ids.remove(wrong_gpu)

        if wrong_gpus:
            print(f"Wrong gpu indeces are excluded. {','.join([str(val) for val in gpu_ids])} GPU will be used.")

        return gpu_ids

    def is_available(self, template) -> bool:
        return len(self._gpu_ids) > 1 and not template.task_type.is_anomaly

    def setup_multi_gpu_train(
        self,
        output_path: str,
        optimized_hyper_parameters: Optional[ConfigurableParameters] = None
    ):
        if optimized_hyper_parameters is not None:
            self._set_optimized_hp_for_child_process(optimized_hyper_parameters)

        self._processes = self._spawn_multi_gpu_processes(output_path)

        signal.signal(signal.SIGINT, self._terminate_signal_handler)
        signal.signal(signal.SIGTERM, self._terminate_signal_handler)

        self.initialize_multigpu_train(0, self._gpu_ids, self._multi_gpu_port)

        t = threading.Thread(target=self._check_child_processes_alive, daemon=True)
        t.start()

    def finalize(self):
        if self._processes is not None:
            for p in self._processes:
                p.join()

    @staticmethod
    def initialize_multigpu_train(rank: int, gpu_ids: List[int], multi_gpu_port: str):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = multi_gpu_port
        torch.cuda.set_device(gpu_ids[rank])
        dist.init_process_group(backend='nccl', world_size=len(gpu_ids), rank=rank)
        print(f'dist info world_size = {dist.get_world_size()}, rank = {dist.get_rank()}')

    @staticmethod
    def run_child_process(rank: int, gpu_ids: List[int], output_path: str, multi_gpu_port: str):
        gpus_arg_idx = sys.argv.index('--gpus')
        for _ in range(2):
            sys.argv.pop(gpus_arg_idx)
        if "--enable-hpo" in sys.argv:
            sys.argv.remove('--enable-hpo')
        MultiGPUManager.set_arguments_to_argv("--save-logs-to", output_path)

        MultiGPUManager.initialize_multigpu_train(rank, gpu_ids, multi_gpu_port)

        main()

    @staticmethod
    def set_arguments_to_argv(key: str, value: str, after_params: bool = False):
        if key in sys.argv:
            sys.argv[sys.argv.index(key) + 1] = value
        else:
            if not after_params and "params" in sys.argv:
                sys.argv.insert(sys.argv.index("params"), key)
                sys.argv.insert(sys.argv.index("params"), value)
            else:
                if after_params and "params" not in sys.argv:
                    sys.argv.append('params')
                sys.argv.extend([key, value])

    def _spawn_multi_gpu_processes(self, output_path: str) -> List[mp.Process]:
        processes= []
        spawned_mp = mp.get_context("spawn")
        for rank in range(1, len(self._gpu_ids)):
            task_p = spawned_mp.Process(
                target=MultiGPUManager.run_child_process,
                args=(rank, self._gpu_ids, output_path, self._multi_gpu_port)
            )
            task_p.start()
            processes.append(task_p)

        return processes

    def _terminate_signal_handler(self, signum, frame):
        # This code prevents child processses from being killed unintentionally by forked main process
        if self._main_pid != os.getpid():
            sys.exit()

        self._kill_child_process()

        singal_name = {2: "SIGINT", 15: "SIGTERM"}
        print(f"{singal_name[signum]} is sent. process exited.")

        sys.exit(1)

    def _kill_child_process(self):
        if self._processes is None:
            return

        for process in self._processes:
            print(f"Kill child process {process.pid}")
            try:
                process.kill()
            except Exception:
                pass

    def _set_optimized_hp_for_child_process(self, hyper_parameters: ConfigurableParameters):
        self.set_arguments_to_argv(
            "--learning_parameters.learning_rate",
            str(hyper_parameters.learning_parameters.learning_rate),
            True
        )
        self.set_arguments_to_argv(
            "--learning_parameters.batch_size",
            str(hyper_parameters.learning_parameters.batch_size),
            True
        )

    def _check_child_processes_alive(self):
        child_is_running = True
        while child_is_running:
            time.sleep(1)
            for p in self._processes:
                if not p.is_alive() and p.exitcode != 0:
                    child_is_running = False
                    break

        print("Some of child processes are terminated abnormally. process exits.")
        self._kill_child_process()
        os.kill(self._main_pid, signal.SIGKILL)


if __name__ == "__main__":
    main()
