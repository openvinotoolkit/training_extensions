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
from ote_cli.utils.config import set_values_as_default
from ote_cli.utils.importing import get_impl_class
from ote_cli.utils.labels import generate_label_schema
from ote_cli.utils.loading import load_model_weights
from ote_sdk.entities.model_template import parse_model_template
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType
from sc_sdk.entities.dataset_storage import NullDatasetStorage
from sc_sdk.entities.datasets import NullDataset
from sc_sdk.entities.model import Model, ModelOptimizationType, ModelPrecision, ModelStatus, TargetDevice
from sc_sdk.entities.model_storage import NullModelStorage
from sc_sdk.entities.project import NullProject
from sc_sdk.logging import logger_factory

from mmdet.integration.nncf import is_checkpoint_nncf


logger = logger_factory.get_logger("Sample")


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


    is_nncf = is_checkpoint_nncf(args.load_weights)
    # Get classes for Task, ConfigurableParameters and Dataset.
    Task = get_impl_class(template.entrypoints.nncf if is_nncf else template.entrypoints.base)
    Dataset = get_dataset_class(template.task_type)

    assert args.labels is not None or args.ann_files is not None

    if args.labels:
        labels = args.labels
    else:
        Dataset = get_dataset_class(template.task_type)
        dataset = Dataset(args.ann_files, dataset_storage=NullDatasetStorage())
        labels = dataset.get_labels()

    labels_schema = generate_label_schema(labels, template.task_type)

    # Get hyper parameters schema.
    hyper_parameters = template.hyper_parameters.data
    assert hyper_parameters
    # Sync values with default values.
    set_values_as_default(hyper_parameters)

    environment = TaskEnvironment(
        model=None,
        hyper_parameters=hyper_parameters,
        label_schema=labels_schema,
        model_template=template)

    model_bytes = load_model_weights(args.load_weights)
    model = Model(project=NullProject(),
                  model_storage=NullModelStorage(),
                  configuration=environment.get_model_configuration(),
                  data_source_dict={'weights.pth': model_bytes},
                  train_dataset=NullDataset())
    environment.model = model

    task = Task(task_environment=environment)

    exported_model = Model(
        NullProject(),
        NullModelStorage(),
        NullDataset(),
        environment.get_model_configuration(),
        optimization_type=ModelOptimizationType.MO,
        precision=[ModelPrecision.FP16],
        optimization_methods=[],
        optimization_objectives={},
        target_device=TargetDevice.UNSPECIFIED,
        performance_improvement={},
        model_size_reduction=1.,
        model_status=ModelStatus.NOT_READY)

    task.export(ExportType.OPENVINO, exported_model)

    os.makedirs(args.save_model_to, exist_ok=True)

    with open(os.path.join(args.save_model_to, 'model.bin'), 'wb') as write_file:
        write_file.write(exported_model.get_data('openvino.bin'))

    with open(os.path.join(args.save_model_to, 'model.xml'), 'w') as write_file:
        write_file.write(exported_model.get_data('openvino.xml').decode())
