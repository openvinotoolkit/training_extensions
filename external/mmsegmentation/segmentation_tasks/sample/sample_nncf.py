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
import logging
import os
import os.path as osp
import sys
import time

from ote_sdk.configuration.helper import create
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.label_schema import LabelSchemaEntity
from ote_sdk.entities.model import ModelEntity
from ote_sdk.entities.model_template import parse_model_template
from ote_sdk.entities.optimization_parameters import OptimizationParameters
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.subset import Subset
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType
from ote_sdk.usecases.tasks.interfaces.optimization_interface import OptimizationType
from ote_sdk.entities.task_environment import TaskEnvironment

from segmentation_tasks.apis.segmentation.ote_utils import get_task_class
from segmentation_tasks.extension.datasets.mmdataset import load_dataset_items
from ote_sdk.usecases.adapters.model_adapter import ModelAdapter


logger = logging.getLogger(__name__)

RESULTS = dict()
TIMES = dict()

def parse_args():
    parser = argparse.ArgumentParser(description='Sample showcasing the new API')
    parser.add_argument('template_file_path', help='path to template file')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--weights', default='weights.pth')
    return parser.parse_args()


def prepare(data_dir, template_file_path):
    logger.info('Initialize dataset')
    labels_list = []
    items = load_dataset_items(
        ann_file_path=osp.join(data_dir, 'annotations/training'),
        data_root_dir=osp.join(data_dir, 'images/training'),
        subset=Subset.TRAINING,
        labels_list=labels_list)
    items.extend(load_dataset_items(
        ann_file_path=osp.join(data_dir, 'annotations/validation'),
        data_root_dir=osp.join(data_dir, 'images/validation'),
        subset=Subset.VALIDATION,
        labels_list=labels_list))
    items.extend(load_dataset_items(
        ann_file_path=osp.join(data_dir, 'annotations/validation'),
        data_root_dir=osp.join(data_dir, 'images/validation'),
        subset=Subset.TESTING,
        labels_list=labels_list))
    dataset = DatasetEntity(items=items)


    labels_schema = LabelSchemaEntity.from_labels(labels_list)

    logger.info(f'Train dataset: {len(dataset.get_subset(Subset.TRAINING))} items')
    logger.info(f'Validation dataset: {len(dataset.get_subset(Subset.VALIDATION))} items')

    logger.info('Load model template')
    model_template = parse_model_template(template_file_path)


    logger.info('Setup environment')
    params = create(model_template.hyper_parameters.data)
    logger.info('Set hyperparameters')
    params.learning_parameters.learning_rate_fixed_iters = 3
    params.learning_parameters.learning_rate_warmup_iters = 3
    params.learning_parameters.num_iters = 5
    environment = TaskEnvironment(model=None,
                                  hyper_parameters=params,
                                  label_schema=labels_schema,
                                  model_template=model_template)

    logger.info('Setup NNCF environment')
    params_nncf = create(model_template.hyper_parameters.data)
    params_nncf.nncf_optimization.maximal_accuracy_degradation = 1.0
    params_nncf.nncf_optimization.enable_quantization = True
    params_nncf.nncf_optimization.enable_pruning = False
    environment_nncf = TaskEnvironment(model=None,
                                  hyper_parameters=params_nncf,
                                  label_schema=labels_schema,
                                  model_template=model_template)
    return dataset, model_template, environment, environment_nncf

def load_model_weights(path):
    with open(path, 'rb') as read_file:
        return read_file.read()

def train(model_template, dataset, environment):
    logger.info('Create base Task')
    task_impl_path = model_template.entrypoints.base
    task_cls = get_task_class(task_impl_path)
    task = task_cls(task_environment=environment)

    logger.info('Train model')
    output_model = ModelEntity(
        dataset,
        environment.get_model_configuration(),
    )
    if environment.model is None:
        task.train(dataset, output_model)
    else:
        print("Skipped training, because weights exists")

    eval_pt(task, dataset, output_model, "FP32")

    export(task, dataset, environment, "FP32")

    return output_model

def eval_pt(task, dataset, model, model_name):
    begin_time = time.time()
    logger.info('Get predictions on the validation set')
    validation_dataset = dataset.get_subset(Subset.VALIDATION)
    predicted_validation_dataset = task.infer(validation_dataset.with_empty_annotations(),
                                              InferenceParameters(is_evaluation=True))
    resultset = ResultSetEntity(model=model,
                                ground_truth_dataset=validation_dataset,
                                prediction_dataset=predicted_validation_dataset)
    logger.info('Estimate quality on validation set')
    task.evaluate(resultset)
    resultset.performance=0
    RESULTS[model_name] = resultset.performance
    TIMES[f"eval_{model_name}"] = time.time() - begin_time
    logger.info(str(resultset.performance))

def export(task, dataset, environment, model_name):
    begin_time = time.time()
    exported_model = ModelEntity(
        dataset,
        environment.get_model_configuration(),
    )
    task.export(ExportType.OPENVINO, exported_model)
    TIMES[f"export_{model_name}"] = time.time() - begin_time

    export_dir = "IR"
    os.makedirs(export_dir, exist_ok=True)
    with open(osp.join(export_dir, model_name + ".bin"), "wb") as f:
        f.write(exported_model.get_data("openvino.bin"))
    with open(osp.join(export_dir, model_name + ".xml"), "wb") as f:
        f.write(exported_model.get_data("openvino.xml"))


def main(args):
    dataset, model_template, environment, environment_nncf = prepare(args.data_dir, args.template_file_path)

    if not osp.exists(args.weights):
        trained_model = train(model_template, dataset, environment)
        with open(args.weights, "wb") as f:
            f.write(trained_model.get_data('weights.pth'))
    else:
        model_bytes = load_model_weights(args.weights)
        trained_model = ModelEntity(
            train_dataset=dataset,
            configuration=environment.get_model_configuration(),
            model_adapters={'weights.pth': ModelAdapter(model_bytes)}
        )
        environment.model = trained_model
        train(model_template, dataset, environment)

    environment_nncf.model = trained_model
    logger.info('Create NNCF Task')
    begin_time = time.time()
    task_cls = get_task_class(model_template.entrypoints.nncf)
    nncf_task = task_cls(task_environment=environment_nncf)

    logger.info('Optimize model')
    output_model = ModelEntity(
        dataset,
        environment.get_model_configuration(),
    )
    optimize_parameters = OptimizationParameters()

    nncf_task.optimize(OptimizationType.NNCF, dataset, output_model, optimize_parameters)
    TIMES[f"optimize"] = time.time() - begin_time

    eval_pt(nncf_task, dataset, output_model, "INT8")
    export(nncf_task, dataset, environment, "INT8")

    print("\nPerformance:")
    for k,v in sorted(RESULTS.items()):
        print(f"{k} : {v:.5f}")

    print("\nTimes:")
    for k,v in sorted(TIMES.items()):
        print(f"{k} : {v:.1f}")

if __name__ == '__main__':
    sys.exit(main(parse_args()) or 0)
