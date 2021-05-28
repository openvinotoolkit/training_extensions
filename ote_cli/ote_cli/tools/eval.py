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

from ote_cli.common import (MODEL_TEMPLATE_FILENAME,
                            add_hyper_parameters_sub_parser, create_project,
                            gen_params_dict_from_args, get_task_impl_class,
                            load_config, load_model_weights)
from ote_cli.datasets import get_dataset_class
from sc_sdk.entities.analyse_parameters import AnalyseParameters
from sc_sdk.entities.datasets import NullDataset, Subset
from sc_sdk.entities.model import Model
from sc_sdk.entities.resultset import ResultSet
from sc_sdk.entities.task_environment import TaskEnvironment
from sc_sdk.logging import logger_factory

logger = logger_factory.get_logger("Sample")


def parse_args(config):
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-ann-files', required=True,
                        help='Comma-separated paths to test annotation files.')
    parser.add_argument('--test-data-roots', required=True,
                        help='Comma-separated paths to test data folders.')
    parser.add_argument('--load-weights', required=True,
                        help='Load only weights from previously saved checkpoint')

    add_hyper_parameters_sub_parser(parser, config)

    return parser.parse_args()


def main():
    config = load_config(MODEL_TEMPLATE_FILENAME)
    args = parse_args(config)
    updated_hyper_parameters = gen_params_dict_from_args(args)
    if updated_hyper_parameters:
        config['hyper_parameters']['params'] = updated_hyper_parameters['params']

    print(updated_hyper_parameters)

    Task = get_task_impl_class(config)
    Dataset = get_dataset_class(config['domain'])

    dataset = Dataset(test_ann_file=args.test_ann_files,
                      test_data_root=args.test_data_roots)

    project = create_project(dataset.get_labels())
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

    dataset.set_project_labels(project.get_labels())

    validation_dataset = dataset.get_subset(Subset.TESTING)
    predicted_validation_dataset = task.analyse(
        validation_dataset.with_empty_annotations(),
        AnalyseParameters(is_evaluation=True))

    resultset = ResultSet(
        model=model,
        ground_truth_dataset=validation_dataset,
        prediction_dataset=predicted_validation_dataset,
    )
    performance = task.compute_performance(resultset)
    resultset.performance = performance
