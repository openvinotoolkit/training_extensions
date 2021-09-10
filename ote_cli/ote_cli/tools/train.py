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
from ote_cli.utils.config import override_parameters, set_values_as_default
from ote_cli.utils.importing import get_impl_class
from ote_cli.utils.labels import generate_label_schema
from ote_cli.utils.loading import load_model_weights
from ote_cli.utils.parser import (add_hyper_parameters_sub_parser,
                                  gen_params_dict_from_args)
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.model_template import parse_model_template
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.task_environment import TaskEnvironment
from sc_sdk.entities.dataset_storage import NullDatasetStorage
from sc_sdk.entities.datasets import NullDataset, Subset
from sc_sdk.entities.model import Model, ModelStatus
from sc_sdk.entities.model_storage import NullModelStorage
from sc_sdk.entities.project import NullProject
from sc_sdk.logging import logger_factory

logger = logger_factory.get_logger("Sample")


def parse_args(config):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-ann-files', required=True,
                        help='Comma-separated paths to training annotation files.')
    parser.add_argument('--train-data-roots', required=True,
                        help='Comma-separated paths to training data folders.')
    parser.add_argument('--val-ann-files', required=True,
                        help='Comma-separated paths to validation annotation files.')
    parser.add_argument('--val-data-roots', required=True,
                        help='Comma-separated paths to validation data folders.')
    parser.add_argument('--load-weights', required=False,
                        help='Load only weights from previously saved checkpoint')
    parser.add_argument('--save-weights', required=True,
                        help='Location to store weights.')

    add_hyper_parameters_sub_parser(parser, config)

    return parser.parse_args()


def main():
    # Load template.yaml file.
    template = parse_model_template('template.yaml')
    # Get hyper parameters schema.
    hyper_parameters = template.hyper_parameters.data
    assert hyper_parameters
    # Sync values with default values.
    set_values_as_default(hyper_parameters)
    # Dynamically create an argument parser based on override parameters.
    args = parse_args(hyper_parameters)
    # Get new values from user's input.
    updated_hyper_parameters = gen_params_dict_from_args(args)
    # Override overridden parameters by user's values.
    override_parameters(updated_hyper_parameters, hyper_parameters, allow_value=True)

    # Get classes for Task, ConfigurableParameters and Dataset.
    Task = get_impl_class(template.entrypoints.base)
    Dataset = get_dataset_class(template.task_type)

    # Create instances of Task, ConfigurableParameters and Dataset.
    dataset = Dataset(train_ann_file=args.train_ann_files,
                      train_data_root=args.train_data_roots,
                      val_ann_file=args.val_ann_files,
                      val_data_root=args.val_data_roots,
                      dataset_storage=NullDatasetStorage())

    labels_schema = generate_label_schema(dataset.get_labels(), template.task_type)
    labels_list = labels_schema.get_labels(False)
    dataset.set_project_labels(labels_list)

    environment = TaskEnvironment(
        model=None,
        hyper_parameters=hyper_parameters,
        label_schema=labels_schema,
        model_template=template)

    if args.load_weights:
        model_bytes = load_model_weights(args.load_weights)
        model = Model(project=NullProject(),
                      model_storage=NullModelStorage(),
                      configuration=environment.get_model_configuration(),
                      data_source_dict={'weights.pth': model_bytes},
                      train_dataset=NullDataset())
        environment.model = model

    task = Task(task_environment=environment)

    output_model = Model(
        NullProject(),
        NullModelStorage(),
        dataset,
        environment.get_model_configuration(),
        model_status=ModelStatus.NOT_READY)

    task.train(dataset, output_model)

    if output_model.model_status != ModelStatus.NOT_READY:
        with open(args.save_weights, 'wb') as f:
            f.write(output_model.get_data("weights.pth"))

    validation_dataset = dataset.get_subset(Subset.VALIDATION)
    predicted_validation_dataset = task.infer(
        validation_dataset.with_empty_annotations(),
        InferenceParameters(is_evaluation=True))

    resultset = ResultSetEntity(
        model=output_model,
        ground_truth_dataset=validation_dataset,
        prediction_dataset=predicted_validation_dataset,
    )
    performance = task.evaluate(resultset)
    print(performance)
