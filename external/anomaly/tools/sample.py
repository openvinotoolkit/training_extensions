"""
Sample to run anomaly detection task via ote_sdk API
"""

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
import importlib
import os
import shutil
import sys
import time
from typing import Callable

from ote_anomalib.data.helpers import OTEAnomalyDatasetGenerator
from ote_sdk.configuration.helper import create
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.label_schema import LabelSchemaEntity
from ote_sdk.entities.model import (
    ModelEntity,
    ModelOptimizationType,
    ModelPrecision,
    ModelStatus,
    OptimizationMethod,
)
from ote_sdk.entities.model_template import TargetDevice, parse_model_template
from ote_sdk.entities.optimization_parameters import OptimizationParameters
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.entities.train_parameters import TrainParameters
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType
from ote_sdk.usecases.tasks.interfaces.optimization_interface import OptimizationType


def get_task_class(path: str) -> Callable:
    """
    Return class by path
    :param path: path to class
    :return: class
    """
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def parse_args():
    """
    Argument parser
    """
    parser = argparse.ArgumentParser(description="Sample showcasing the new API")
    parser.add_argument("template_file_path", help="path to template file")
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--save-dir", default=".")
    return parser.parse_args()


# pylint: disable=too-many-locals, too-many-statements, redefined-outer-name, unspecified-encoding
def main(args):
    """
    Sample to run anomaly detection task via ote_sdk API
    :param args:
    """
    print("Load model template")
    model_template = parse_model_template(args.template_file_path)
    print(model_template)

    category_data_dir = os.path.join(args.data_dir, args.category)
    dataset_generator = OTEAnomalyDatasetGenerator(category_data_dir, seed=777, create_validation_set=True)
    dataset = dataset_generator.generate()

    print(f"Train dataset: {len(dataset.get_subset(Subset.TRAINING))} items")
    print(f"Validation dataset: {len(dataset.get_subset(Subset.VALIDATION))} items")
    print(f"Test dataset: {len(dataset.get_subset(Subset.TESTING))} items")

    hyper_parameters = create(input_config=model_template.hyper_parameters.data)

    labels = [dataset_generator.normal_label, dataset_generator.abnormal_label]
    label_schema = LabelSchemaEntity.from_labels(labels)

    task_environment = TaskEnvironment(
        model_template=model_template,
        model=None,
        hyper_parameters=hyper_parameters,
        label_schema=label_schema,
    )

    print("Train model")

    task_cls = get_task_class(model_template.entrypoints.base)
    task = task_cls(task_environment=task_environment)

    output_model = ModelEntity(
        dataset,
        task_environment.get_model_configuration(),
        model_status=ModelStatus.NOT_READY,
    )

    start = time.time()
    task.train(dataset, output_model, TrainParameters())
    time_train = time.time() - start

    #############################################

    print("Export model")
    fp32_model = ModelEntity(
        dataset,
        task_environment.get_model_configuration(),
        model_status=ModelStatus.NOT_READY,
    )

    start = time.time()
    task.export(ExportType.OPENVINO, fp32_model)
    time_export = time.time() - start

    task_cls = get_task_class(model_template.entrypoints.openvino)
    task_environment.model = fp32_model
    openvino_task = task_cls(task_environment=task_environment)

    inference_dataset = dataset.get_subset(Subset.TESTING)
    inference_parameters = InferenceParameters(is_evaluation=True)

    hyper_parameters = task_environment.get_hyper_parameters()
    openvino_task.task_environment.set_hyper_parameters(hyper_parameters=hyper_parameters)

    predicted_fp32_dataset = openvino_task.infer(
        dataset=inference_dataset.with_empty_annotations(),
        inference_parameters=inference_parameters,
    )

    result_fp32_set = ResultSetEntity(
        model=fp32_model,
        ground_truth_dataset=inference_dataset,
        prediction_dataset=predicted_fp32_dataset,
    )

    openvino_task.evaluate(output_resultset=result_fp32_set)
    perf_fp32 = result_fp32_set.performance
    perf_fp32_value = perf_fp32.score.value
    print(f"Training FP32 performance: {perf_fp32.score.name}, {perf_fp32_value:.4f}")

    #################################################

    print("Optimize model")
    optimized_model = ModelEntity(
        inference_dataset,
        task_environment.get_model_configuration(),
        optimization_type=ModelOptimizationType.POT,
        optimization_methods=[OptimizationMethod.QUANTIZATION],
        optimization_objectives={},
        precision=[ModelPrecision.INT8],
        target_device=TargetDevice.CPU,
        performance_improvement={},
        model_size_reduction=1.0,
        model_status=ModelStatus.NOT_READY,
    )

    start = time.time()

    openvino_task.optimize(
        optimization_type=OptimizationType.POT,
        dataset=inference_dataset,
        output_model=optimized_model,
        optimization_parameters=OptimizationParameters(),
    )

    time_pot = time.time() - start

    predicted_inference_int8_dataset = openvino_task.infer(
        dataset=inference_dataset.with_empty_annotations(),
        inference_parameters=inference_parameters,
    )
    result_int8_set = ResultSetEntity(
        model=optimized_model,
        ground_truth_dataset=inference_dataset,
        prediction_dataset=predicted_inference_int8_dataset,
    )

    openvino_task.evaluate(output_resultset=result_int8_set)
    perf_int8 = result_int8_set.performance
    perf_int8_value = perf_int8.score.value
    print(f"Training int8 performance: {perf_int8.score.name}, {perf_int8_value:.4f}")

    ################

    result_txt = f"category;{args.category};"
    result_txt += f"fp32;{perf_fp32_value};"
    result_txt += f"int8;{perf_int8_value};"
    result_txt += f"time_train;{time_train};"
    result_txt += f"time_export;{time_export};"
    result_txt += f"time_pot;{time_pot}\n"

    model_name = "padim" if "padim" in args.template_file_path else "stfpm"
    os.makedirs(model_name, exist_ok=True)
    with open(os.path.join(model_name, args.category + ".log"), "w") as f:
        f.write(result_txt)

    os.makedirs(os.path.join(model_name, args.category), exist_ok=True)
    save_dir = os.path.join(args.save_dir, model_name, args.category)
    with open(os.path.join(save_dir, "fp32.bin"), "wb") as f:
        f.write(fp32_model.get_data("openvino.bin"))
    with open(os.path.join(save_dir, "fp32.xml"), "wb") as f:
        f.write(fp32_model.get_data("openvino.xml"))

    with open(os.path.join(save_dir, "int8.bin"), "wb") as f:
        f.write(optimized_model.get_data("openvino.bin"))
    with open(os.path.join(save_dir, "int8.xml"), "wb") as f:
        f.write(optimized_model.get_data("openvino.xml"))

    # Clean dir, that used by anomalib
    folder_path = "results"
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)


# pylint: disable=redefined-outer-name
if __name__ == "__main__":
    args = parse_args()
    if args.category != "all":
        main(args)
        sys.exit(0)

    categories = [name for name in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, name))]
    num_categories = len(categories)
    for i, category in enumerate(sorted(categories)):
        args.category = category
        print("--------------")
        print(f"category[{i+1}/{num_categories}]: {category}")
        main(args)
