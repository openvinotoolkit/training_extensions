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

from ote_cli.datasets import get_dataset_class
from ote_cli.utils.config import apply_template_configurable_parameters
from ote_cli.utils.importing import get_task_impl_class
from ote_cli.utils.labels import generate_label_schema
from ote_cli.utils.loading import load_config, load_model_weights
from ote_cli.utils.parser import (add_hyper_parameters_sub_parser,
                                  gen_params_dict_from_args)
from sc_sdk.entities.dataset_storage import NullDatasetStorage
from sc_sdk.entities.datasets import NullDataset, Subset
from sc_sdk.entities.id import ID
from sc_sdk.entities.inference_parameters import InferenceParameters
from sc_sdk.entities.model import Model, NullModel
from sc_sdk.entities.model_storage import NullModelStorage
from sc_sdk.entities.project import NullProject
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
    # Load template.yaml file.
    template = load_config()

    # Dynamically create an argument parser based on loaded template.yaml file.
    args = parse_args(template)
    updated_hyper_parameters = gen_params_dict_from_args(args)
    if updated_hyper_parameters:
        template['hyper_parameters']['params'] = updated_hyper_parameters['params']

    # Get classes for Task, ConfigurableParameters and Dataset.
    Task = get_task_impl_class(template['task']['base'])
    ConfigurableParameters = get_task_impl_class(template['hyper_parameters']['impl'])
    Dataset = get_dataset_class(template['domain'])

    dataset = Dataset(test_ann_file=args.test_ann_files,
                      test_data_root=args.test_data_roots,
                      dataset_storage=NullDatasetStorage())

    params = ConfigurableParameters(workspace_id=ID(), project_id=ID(), task_id=ID())
    apply_template_configurable_parameters(params, template)

    labels_schema = generate_label_schema(dataset.get_labels(), template['domain'])
    labels_list = labels_schema.get_labels(False)
    dataset.set_project_labels(labels_list)

    environment = TaskEnvironment(model=NullModel(), configurable_parameters=params, label_schema=labels_schema)

    model_bytes = load_model_weights(args.load_weights)
    model = Model(project=NullProject(),
                  model_storage=NullModelStorage(),
                  configuration=environment.get_model_configuration(),
                  data_source_dict={'weights.pth': model_bytes},
                  train_dataset=NullDataset())
    environment.model = model

    task = Task(task_environment=environment)

    validation_dataset = dataset.get_subset(Subset.TESTING)
    predicted_validation_dataset = task.infer(
        validation_dataset.with_empty_annotations(),
        InferenceParameters(is_evaluation=True))

    resultset = ResultSet(
        model=model,
        ground_truth_dataset=validation_dataset,
        prediction_dataset=predicted_validation_dataset,
    )
    performance = task.evaluate(resultset)
    print(performance)
