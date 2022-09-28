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
import os.path as osp
import sys
import time

from ote_sdk.configuration.helper import create
from ote_sdk.entities.datasets import Subset
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.model import ModelEntity, ModelPrecision, ModelOptimizationType
from ote_sdk.entities.model_template import parse_model_template
from ote_sdk.entities.optimization_parameters import OptimizationParameters
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.usecases.adapters.model_adapter import ModelAdapter
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType
from ote_sdk.usecases.tasks.interfaces.optimization_interface import OptimizationType

from torchreid.integration.nncf.compression import is_nncf_checkpoint
from torchreid_tasks.utils import (ClassificationDatasetAdapter,
                                   get_task_class)

def parse_args():
    parser = argparse.ArgumentParser(description='Sample showcasing the new API')
    parser.add_argument('template_file_path', help='path to template file')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--weights',
                        help='path to the pre-trained model weights',
                        default=None)
    parser.add_argument('--aux-weights',
                        help='path to the pre-trained aux model weights',
                        default=None)
    parser.add_argument('--optimize', choices=['nncf', 'pot', 'none'], default='pot')
    parser.add_argument('--enable_quantization', action='store_true')
    parser.add_argument('--enable_pruning', action='store_true')
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--debug-dump-folder', default='')
    args = parser.parse_args()
    return args


def load_weights(path):
    with open(path, 'rb') as read_file:
        return read_file.read()


def validate(task, validation_dataset, model):
    print('Get predictions on the validation set')
    predicted_validation_dataset = task.infer(
        validation_dataset.with_empty_annotations(),
        InferenceParameters(is_evaluation=True))
    resultset = ResultSetEntity(
        model=model,
        ground_truth_dataset=validation_dataset,
        prediction_dataset=predicted_validation_dataset,
    )
    print('Estimate quality on validation set')
    task.evaluate(resultset)
    print(str(resultset.performance))


def main(args):
    if args.debug_dump_folder:
        from torchreid.utils import Logger
        log_name = 'ote_task.log' + time.strftime('-%Y-%m-%d-%H-%M-%S')
        sys.stdout = Logger(osp.join(args.debug_dump_folder, log_name))
    print('Initialize dataset')
    dataset = ClassificationDatasetAdapter(
        train_data_root=osp.join(args.data_dir, 'train'),
        train_ann_file=osp.join(args.data_dir, 'train.json'),
        val_data_root=osp.join(args.data_dir, 'val'),
        val_ann_file=osp.join(args.data_dir, 'val.json'),
        test_data_root=osp.join(args.data_dir, 'val'),
        test_ann_file=osp.join(args.data_dir, 'val.json'))

    labels_schema = dataset.generate_label_schema()
    print(f'Train dataset: {len(dataset.get_subset(Subset.TRAINING))} items')
    print(f'Validation dataset: {len(dataset.get_subset(Subset.VALIDATION))} items')

    print('Load model template')
    model_template = parse_model_template(args.template_file_path)

    print('Set hyperparameters')
    params = create(model_template.hyper_parameters.data)
    params.nncf_optimization.enable_quantization = args.enable_quantization
    params.nncf_optimization.enable_pruning = args.enable_pruning

    print('Setup environment')
    environment = TaskEnvironment(model=None,
                                  hyper_parameters=params,
                                  label_schema=labels_schema,
                                  model_template=model_template)

    validation_dataset = dataset.get_subset(Subset.VALIDATION)

    if args.weights is None:
        trained_model = ModelEntity(
            train_dataset=dataset,
            configuration=environment.get_model_configuration(),
        )

        print('Create base Task')
        task_impl_path = model_template.entrypoints.base
        task_cls = get_task_class(task_impl_path)
        task = task_cls(task_environment=environment)

        print('Train model')
        task.train(dataset, trained_model)

        validate(task, validation_dataset, trained_model)
    else:
        print('Load pre-trained weights')
        if is_nncf_checkpoint(args.weights):
            task_impl_path = model_template.entrypoints.nncf
            optimization_type = ModelOptimizationType.NNCF
        else:
            task_impl_path = model_template.entrypoints.base
            optimization_type = ModelOptimizationType.NONE

        weights = load_weights(args.weights)
        model_adapters = {'weights.pth': ModelAdapter(weights)}
        if args.aux_weights is not None:
            aux_weights = load_weights(args.aux_weights)
            model_adapters['aux_model_1.pth'] = ModelAdapter(aux_weights)
        trained_model = ModelEntity(
            train_dataset=dataset,
            configuration=environment.get_model_configuration(),
            model_adapters=model_adapters,
            precision = [ModelPrecision.FP32],
            optimization_type=optimization_type
        )
        environment.model = trained_model

        task_name = task_impl_path.split('.')[-1]
        print(f'Create {task_name} Task')
        task_cls = get_task_class(task_impl_path)
        task = task_cls(task_environment=environment)

        validate(task, validation_dataset, trained_model)

    if args.optimize == 'nncf':
        task_impl_path = model_template.entrypoints.nncf
        nncf_task_cls = get_task_class(task_impl_path)
        if not isinstance(task, nncf_task_cls):
            print('Create NNCF Task')
            environment.model = trained_model
            task = nncf_task_cls(task_environment=environment)

            validate(task, validation_dataset, trained_model)

        print('Optimize model')
        output_model = ModelEntity(
            dataset,
            environment.get_model_configuration(),
        )
        task.optimize(OptimizationType.NNCF, dataset, output_model, None)

        validate(task, validation_dataset, output_model)

    if args.export:
        print('Export model')
        exported_model = ModelEntity(
            dataset,
            environment.get_model_configuration(),
        )
        task.export(ExportType.OPENVINO, exported_model)

        print('Create OpenVINO Task')
        environment.model = exported_model
        openvino_task_impl_path = model_template.entrypoints.openvino
        openvino_task_cls = get_task_class(openvino_task_impl_path)
        openvino_task = openvino_task_cls(environment)

        validate(openvino_task, validation_dataset, exported_model)

        if args.optimize == 'pot':
            print('Run POT optimization')
            optimized_model = ModelEntity(
                dataset,
                environment.get_model_configuration(),
            )
            openvino_task.optimize(
                OptimizationType.POT,
                dataset.get_subset(Subset.TRAINING),
                optimized_model,
                OptimizationParameters())

            validate(openvino_task, validation_dataset, optimized_model)


if __name__ == '__main__':
    cl_args = parse_args()
    print(cl_args)
    sys.exit(main(cl_args) or 0)
