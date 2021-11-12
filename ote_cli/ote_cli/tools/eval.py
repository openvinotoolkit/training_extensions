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
from ote_cli.utils.config import override_parameters
from ote_cli.utils.importing import get_impl_class
from ote_cli.utils.loading import load_model_weights
from ote_cli.utils.parser import (add_hyper_parameters_sub_parser,
                                  gen_params_dict_from_args)
from ote_sdk.configuration.helper import create
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.label_schema import LabelSchemaEntity
from ote_sdk.entities.model import ModelEntity
from ote_sdk.entities.model_template import parse_model_template
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.usecases.adapters.model_adapter import ModelAdapter


def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('template')
    parsed, _ = pre_parser.parse_known_args()
    # Load template.yaml file.
    template = parse_model_template(parsed.template)
    # Get hyper parameters schema.
    hyper_parameters = template.hyper_parameters.data
    assert hyper_parameters

    parser = argparse.ArgumentParser()
    parser.add_argument('template')
    parser.add_argument('--test-ann-files', required=True,
                        help='Comma-separated paths to test annotation files.')
    parser.add_argument('--test-data-roots', required=True,
                        help='Comma-separated paths to test data folders.')
    parser.add_argument('--load-weights', required=True,
                        help='Load only weights from previously saved checkpoint')

    add_hyper_parameters_sub_parser(parser, hyper_parameters, modes=('INFERENCE', ))

    return parser.parse_args(), template, hyper_parameters


def main():
    # Dynamically create an argument parser based on override parameters.
    args, template, hyper_parameters = parse_args()
    # Get new values from user's input.
    updated_hyper_parameters = gen_params_dict_from_args(args)
    # Override overridden parameters by user's values.
    override_parameters(updated_hyper_parameters, hyper_parameters, allow_value=True)

    hyper_parameters = create(hyper_parameters)

    # Get classes for Task, ConfigurableParameters and Dataset.
    Task = get_impl_class(template.entrypoints.base)
    Dataset = get_dataset_class(template.task_type)

    dataset = Dataset(test_ann_file=args.test_ann_files,
                      test_data_root=args.test_data_roots)

    labels_schema = LabelSchemaEntity.from_labels(dataset.get_labels())

    environment = TaskEnvironment(
        model=None,
        hyper_parameters=hyper_parameters,
        label_schema=labels_schema,
        model_template=template)

    model_bytes = load_model_weights(args.load_weights)

    model_adapters = {'weights.pth': ModelAdapter(model_bytes)}
    model = ModelEntity(configuration=environment.get_model_configuration(),
                        model_adapters=model_adapters,
                        train_dataset=None)
    environment.model = model

    task = Task(task_environment=environment)

    validation_dataset = dataset.get_subset(Subset.TESTING)
    predicted_validation_dataset = task.infer(
        validation_dataset.with_empty_annotations(),
        InferenceParameters(is_evaluation=True))

    resultset = ResultSetEntity(
        model=model,
        ground_truth_dataset=validation_dataset,
        prediction_dataset=predicted_validation_dataset,
    )
    task.evaluate(resultset)
    assert resultset.performance is not None
    print(resultset.performance)
