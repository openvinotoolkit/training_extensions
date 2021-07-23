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
from ote_cli.utils.importing import get_impl_class
from ote_cli.utils.labels import generate_label_schema
from ote_cli.utils.loading import load_config, load_model_weights
from ote_cli.utils.parser import (add_hyper_parameters_sub_parser,
                                  gen_params_dict_from_args)
from sc_sdk.entities.dataset_storage import NullDatasetStorage
from sc_sdk.entities.datasets import NullDataset, Subset
from sc_sdk.entities.id import ID
from sc_sdk.entities.inference_parameters import InferenceParameters
from sc_sdk.entities.model import Model, ModelStatus, NullModel
from sc_sdk.entities.model_storage import NullModelStorage
from sc_sdk.entities.project import NullProject
from sc_sdk.entities.resultset import ResultSet
from sc_sdk.entities.task_environment import TaskEnvironment
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
                        help='Location to store wiehgts.')

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
    Task = get_impl_class(template['task']['base'])
    ConfigurableParameters = get_impl_class(template['hyper_parameters']['impl'])
    Dataset = get_dataset_class(template['domain'])

    # Create instances of Task, ConfigurableParameters and Dataset.
    dataset = Dataset(train_ann_file=args.train_ann_files,
                      train_data_root=args.train_data_roots,
                      val_ann_file=args.val_ann_files,
                      val_data_root=args.val_data_roots,
                      dataset_storage=NullDatasetStorage())

    params = ConfigurableParameters(workspace_id=ID(), project_id=ID(), task_id=ID())
    apply_template_configurable_parameters(params, template)

    labels_schema = generate_label_schema(dataset.get_labels(), template['domain'])
    labels_list = labels_schema.get_labels(False)
    dataset.set_project_labels(labels_list)

    environment = TaskEnvironment(model=NullModel(), configurable_parameters=params, label_schema=labels_schema)

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

    resultset = ResultSet(
        model=output_model,
        ground_truth_dataset=validation_dataset,
        prediction_dataset=predicted_validation_dataset,
    )
    performance = task.evaluate(resultset)
    print(performance)
