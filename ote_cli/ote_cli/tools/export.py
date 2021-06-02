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

from ote_cli.common import (MODEL_TEMPLATE_FILENAME, create_project,
                            get_task_impl_class, load_config,
                            load_model_weights)
from ote_cli.datasets import get_dataset_class
from sc_sdk.entities.datasets import NullDataset
from sc_sdk.entities.model import Model
from sc_sdk.entities.task_environment import TaskEnvironment
from sc_sdk.logging import logger_factory
from sc_sdk.usecases.adapters.binary_interpreters import RAWBinaryInterpreter
from sc_sdk.usecases.repos import BinaryRepo

logger = logger_factory.get_logger("Sample")


def parse_args(config):
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-weights', required=True,
                        help='Load only weights from previously saved checkpoint')
    parser.add_argument('--save-model-to', required='True',
                        help='Location where exported model will be stored.')
    parser.add_argument('--ann-files')
    parser.add_argument('--labels', nargs='+')

    return parser.parse_args()


def main():
    config = load_config(MODEL_TEMPLATE_FILENAME)
    args = parse_args(config)

    assert args.labels is not None or args.ann_files is not None

    Task = get_task_impl_class(config)

    if args.labels:
        labels = args.labels
    else:
        Dataset = get_dataset_class(config['domain'])
        dataset = Dataset(args.ann_files)
        labels = dataset.get_labels()

    project = create_project(labels)

    environment = TaskEnvironment(project=project, task_node=project.tasks[-1])

    model_bytes = load_model_weights(args.load_weights)
    model = Model(project=environment.project,
                  task_node=environment.task_node,
                  configuration=environment.get_model_configuration(),
                  data=model_bytes,
                  train_dataset=NullDataset())
    environment.model = model

    params = Task.get_configurable_parameters(environment)
    Task.apply_template_configurable_parameters(params, config)
    params.algo_backend.template.value = MODEL_TEMPLATE_FILENAME
    environment.set_configurable_parameters(params)
    task = Task(task_environment=environment)

    optimized_model = task.optimize_loaded_model()[0]

    binary_interpreter = RAWBinaryInterpreter()
    openvino_xml_data = BinaryRepo(project).get_by_url(optimized_model.openvino_xml_url,
                                                       binary_interpreter=binary_interpreter)
    openvino_bin_data = BinaryRepo(project).get_by_url(optimized_model.openvino_bin_url,
                                                       binary_interpreter=binary_interpreter)

    os.makedirs(args.save_model_to, exist_ok=True)

    with open(os.path.join(args.save_model_to, 'model.bin'), 'wb') as write_file:
        write_file.write(openvino_bin_data)

    with open(os.path.join(args.save_model_to, 'model.xml'), 'w') as write_file:
        write_file.write(openvino_xml_data.decode())

