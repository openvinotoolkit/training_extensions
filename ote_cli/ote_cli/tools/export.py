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
from ote_cli.utils.config import apply_template_configurable_parameters
from ote_cli.utils.importing import get_impl_class
from ote_cli.utils.labels import generate_label_schema
from ote_cli.utils.loading import load_config, load_model_weights
from sc_sdk.entities.datasets import NullDataset
from sc_sdk.entities.id import ID
from sc_sdk.entities.model import Model, ModelStatus, NullModel
from sc_sdk.entities.model_storage import NullModelStorage
from sc_sdk.entities.optimized_model import (ModelOptimizationType,
                                             ModelPrecision, OptimizedModel,
                                             TargetDevice)
from sc_sdk.entities.project import NullProject
from sc_sdk.entities.task_environment import TaskEnvironment
from sc_sdk.logging import logger_factory
from sc_sdk.usecases.tasks.interfaces.export_interface import ExportType

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
    template = load_config()

    args = parse_args()

    # Get classes for Task, ConfigurableParameters and Dataset.
    Task = get_impl_class(template['task']['base'])
    ConfigurableParameters = get_impl_class(template['hyper_parameters']['impl'])
    Dataset = get_dataset_class(template['domain'])

    assert args.labels is not None or args.ann_files is not None

    if args.labels:
        labels = args.labels
    else:
        Dataset = get_dataset_class(template['domain'])
        dataset = Dataset(args.ann_files)
        labels = dataset.get_labels()

    params = ConfigurableParameters(workspace_id=ID(), project_id=ID(), task_id=ID())
    apply_template_configurable_parameters(params, template)

    labels_schema = generate_label_schema(labels, template['domain'])

    environment = TaskEnvironment(model=NullModel(), configurable_parameters=params, label_schema=labels_schema)

    model_bytes = load_model_weights(args.load_weights)
    model = Model(project=NullProject(),
                  model_storage=NullModelStorage(),
                  configuration=environment.get_model_configuration(),
                  data_source_dict={'weights.pth': model_bytes},
                  train_dataset=NullDataset())
    environment.model = model

    task = Task(task_environment=environment)

    exported_model = OptimizedModel(
        NullProject(),
        NullModelStorage(),
        NullDataset(),
        environment.get_model_configuration(),
        ModelOptimizationType.MO,
        [ModelPrecision.FP16],
        optimization_methods=[],
        optimization_level={},
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

