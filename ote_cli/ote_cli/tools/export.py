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

from ote_cli.datasets import get_dataset_class
from ote_cli.utils.importing import get_impl_class
from ote_cli.utils.loading import load_model_weights
from ote_sdk.configuration.helper import create
from ote_sdk.entities.id import ID
from ote_sdk.entities.label import LabelEntity
from ote_sdk.entities.label_schema import LabelSchemaEntity
from ote_sdk.entities.model import ModelEntity, ModelStatus
from ote_sdk.entities.model_template import parse_model_template
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.usecases.adapters.model_adapter import ModelAdapter
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-weights', required=True,
                        help='Load only weights from previously saved checkpoint')
    parser.add_argument('--save-model-to', required='True',
                        help='Location where exported model will be stored.')
    parser.add_argument('--ann-files')
    parser.add_argument('--labels', nargs='+')

    return parser.parse_args()


def main():
    # Load template.yaml file.
    template = parse_model_template('template.yaml')

    args = parse_args()

    # Get classes for Task, ConfigurableParameters and Dataset.
    Task = get_impl_class(template.entrypoints.base)
    Dataset = get_dataset_class(template.task_type)

    assert args.labels is not None or args.ann_files is not None

    if args.labels:
        labels = [LabelEntity(l, template.task_type, id=ID(f'{i}')) for i, l in enumerate(args.labels)]
    else:
        Dataset = get_dataset_class(template.task_type)
        dataset = Dataset(args.ann_files)
        labels = dataset.get_labels()

    labels_schema = LabelSchemaEntity.from_labels(labels)

    # Get hyper parameters schema.
    hyper_parameters = create(template.hyper_parameters.data)
    assert hyper_parameters

    environment = TaskEnvironment(
        model=None,
        hyper_parameters=hyper_parameters,
        label_schema=labels_schema,
        model_template=template)

    model_bytes = load_model_weights(args.load_weights)
    model_adapters = {'weights.pth': ModelAdapter(model_bytes)}
    model = ModelEntity(configuration=environment.get_model_configuration(),
                        model_adapters=model_adapters, train_dataset=None)
    environment.model = model

    task = Task(task_environment=environment)

    exported_model = ModelEntity(
        None,
        environment.get_model_configuration(),
        model_status=ModelStatus.NOT_READY)

    task.export(ExportType.OPENVINO, exported_model)

    os.makedirs(args.save_model_to, exist_ok=True)

    with open(os.path.join(args.save_model_to, 'model.bin'), 'wb') as write_file:
        write_file.write(exported_model.get_data('openvino.bin'))

    with open(os.path.join(args.save_model_to, 'model.xml'), 'w') as write_file:
        write_file.write(exported_model.get_data('openvino.xml').decode())
