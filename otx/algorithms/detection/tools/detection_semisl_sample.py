"""Sample Code of otx training for detection."""

# Copyright (C) 2021-2022 Intel Corporation
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
import sys
from random import randint

import numpy as np
from mmcv.utils import get_logger

from otx.algorithms.common.utils import get_task_class
from otx.api.configuration.helper import create
from otx.api.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
    NullAnnotationSceneEntity,
)
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.image import Image
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.label_schema import LabelSchemaEntity
from otx.api.entities.model import ModelEntity
from otx.api.entities.model_template import parse_model_template
from otx.api.entities.optimization_parameters import OptimizationParameters
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.subset import Subset
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from otx.api.usecases.tasks.interfaces.optimization_interface import OptimizationType

logger = get_logger(name="mmdet")


def parse_args():
    """Parse function for getting model template & check export."""
    parser = argparse.ArgumentParser(description="Sample showcasing the new API")
    parser.add_argument("template_file_path", help="path to template file")
    parser.add_argument("--export", action="store_true")
    return parser.parse_args()


colors = dict(red=(255, 0, 0), green=(0, 255, 0))


def load_test_dataset():
    """Load Sample dataset for detection."""

    def gen_image(resolution, x1, y1, x2, y2, color):
        width, height = resolution
        image = np.full([height, width, 3], fill_value=255, dtype=np.uint8)
        image[int(y1 * height) : int(y2 * height), int(x1 * width) : int(x2 * width), :] = np.full(
            [int(height * y2) - int(height * y1), int(width * x2) - int(width * x1), 3],
            fill_value=colors[color],
            dtype=np.uint8,
        )
        return (image, Rectangle(x1=x1, y1=y1, x2=x2, y2=y2))

    labels = [
        LabelEntity(name="red", domain=Domain.DETECTION, id=0),  # OLD class
        LabelEntity(name="green", domain=Domain.DETECTION, id=1),
    ]

    def get_image(subset, label_id):
        def get_randcoord():
            # disable B311 random - used for the random sampling not for security/crypto
            x1 = randint(0, 9)  # nosec B311
            y1 = randint(0, 9)  # nosec B311
            x2 = min(x1 + 2, 10)
            y2 = min(y1 + 2, 10)
            coord = (x1 / 10, y1 / 10, x2 / 10, y2 / 10)
            return coord

        coord = get_randcoord()
        image, bbox = gen_image((640, 480), *coord, labels[label_id].name)
        if subset != Subset.UNLABELED:
            return DatasetItemEntity(
                media=Image(data=image),
                annotation_scene=AnnotationSceneEntity(
                    annotations=[Annotation(bbox, labels=[ScoredLabel(label=labels[label_id])])],
                    kind=AnnotationSceneKind.ANNOTATION,
                ),
                subset=subset,
            )
        return DatasetItemEntity(
            media=Image(data=image),
            annotation_scene=NullAnnotationSceneEntity(),
            subset=subset,
        )

    train = [get_image(Subset.TRAINING, 0) for i in range(10)]
    train += [get_image(Subset.TRAINING, 1) for i in range(10)]
    val = [get_image(Subset.VALIDATION, 0) for i in range(2)]
    val += [get_image(Subset.VALIDATION, 1) for i in range(2)]
    val += [get_image(Subset.TESTING, 0) for i in range(2)]
    val += [get_image(Subset.TESTING, 1) for i in range(2)]
    unlabeled = [get_image(Subset.UNLABELED, 0) for i in range(100)]
    unlabeled += [get_image(Subset.UNLABELED, 1) for i in range(100)]

    return DatasetEntity(train + val + unlabeled), labels


# pylint: disable=too-many-locals, too-many-statements
def main(args):
    """Main function of Detection Sample."""
    logger.info("[Semi-SL] Train  model with unlabeled dataset")
    dataset, labels_list = load_test_dataset()
    labels_schema = LabelSchemaEntity.from_labels(labels_list)

    logger.info(f"Train dataset: {len(dataset.get_subset(Subset.TRAINING))} items")
    logger.info(f"Validation dataset: {len(dataset.get_subset(Subset.VALIDATION))} items")

    logger.info("Load model template")
    model_template = parse_model_template(args.template_file_path)

    logger.info("Set hyperparameters")
    params = create(model_template.hyper_parameters.data)
    params.learning_parameters.num_iters = 5
    params.learning_parameters.learning_rate = 0.01
    params.learning_parameters.learning_rate_warmup_iters = 1
    params.learning_parameters.batch_size = 4

    logger.info("Setup environment")
    environment = TaskEnvironment(
        model=None,
        hyper_parameters=params,
        label_schema=labels_schema,
        model_template=model_template,
    )

    logger.info("Create base Task")
    task_impl_path = model_template.entrypoints.base
    task_cls = get_task_class(task_impl_path)
    task = task_cls(task_environment=environment)

    logger.info("Train model")
    output_model = ModelEntity(
        dataset,
        environment.get_model_configuration(),
    )
    task.train(dataset, output_model)

    logger.info("Get predictions on the validation set")
    validation_dataset = dataset.get_subset(Subset.VALIDATION)
    predicted_validation_dataset = task.infer(
        validation_dataset.with_empty_annotations(),
        InferenceParameters(is_evaluation=False),
    )
    resultset = ResultSetEntity(
        model=output_model,
        ground_truth_dataset=validation_dataset,
        prediction_dataset=predicted_validation_dataset,
    )
    logger.info("Estimate quality on validation set")
    task.evaluate(resultset)
    logger.info(str(resultset.performance))

    if args.export:
        logger.info("Export model")
        exported_model = ModelEntity(
            dataset,
            environment.get_model_configuration(),
        )
        task.export(ExportType.OPENVINO, exported_model)

        logger.info("Create OpenVINO Task")
        environment.model = exported_model
        openvino_task_impl_path = model_template.entrypoints.openvino
        openvino_task_cls = get_task_class(openvino_task_impl_path)
        openvino_task = openvino_task_cls(environment)

        logger.info("Get predictions on the validation set")
        predicted_validation_dataset = openvino_task.infer(
            validation_dataset.with_empty_annotations(),
            InferenceParameters(is_evaluation=True),
        )

        resultset = ResultSetEntity(
            model=output_model,
            ground_truth_dataset=validation_dataset,
            prediction_dataset=predicted_validation_dataset,
        )
        logger.info("Estimate quality on validation set")
        openvino_task.evaluate(resultset)
        logger.info(str(resultset.performance))

        logger.info("Run POT optimization")
        optimized_model = ModelEntity(
            dataset,
            environment.get_model_configuration(),
        )
        openvino_task.optimize(
            OptimizationType.POT,
            dataset.get_subset(Subset.TRAINING),
            optimized_model,
            OptimizationParameters(),
        )

        logger.info("Get predictions on the validation set")
        predicted_validation_dataset = openvino_task.infer(
            validation_dataset.with_empty_annotations(),
            InferenceParameters(is_evaluation=True),
        )
        resultset = ResultSetEntity(
            model=optimized_model,
            ground_truth_dataset=validation_dataset,
            prediction_dataset=predicted_validation_dataset,
        )
        logger.info("Performance of optimized model:")
        openvino_task.evaluate(resultset)
        logger.info(str(resultset.performance))

        logger.info("Running the NNCF optimization")
        environment.model = output_model
        nncf_task_impl_path = model_template.entrypoints.nncf
        nncf_task_cls = get_task_class(nncf_task_impl_path)
        nncf_task = nncf_task_cls(environment)

        optimized_model = ModelEntity(
            dataset,
            configuration=environment.get_model_configuration(),
        )
        nncf_task.optimize(OptimizationType.NNCF, dataset, optimized_model)

        logger.info("Inferring the optimised model on the validation set.")
        predicted_validation_dataset = nncf_task.infer(
            validation_dataset.with_empty_annotations(),
            InferenceParameters(is_evaluation=True),
        )
        resultset = ResultSetEntity(
            model=optimized_model,
            ground_truth_dataset=validation_dataset,
            prediction_dataset=predicted_validation_dataset,
        )

        logger.info("Evaluating the optimized model on the validation set.")
        nncf_task.evaluate(resultset)
        logger.info(str(resultset.performance))

        logger.info("Exporting the model.")
        exported_model = ModelEntity(
            train_dataset=dataset,
            configuration=environment.get_model_configuration(),
        )
        nncf_task.export(ExportType.OPENVINO, exported_model)
        environment.model = exported_model

        logger.info("Creating the OpenVINO Task.")
        environment.model = exported_model
        openvino_task_impl_path = model_template.entrypoints.openvino
        nncf_openvino_task_cls = get_task_class(openvino_task_impl_path)
        nncf_openvino_task = nncf_openvino_task_cls(environment)

        logger.info("Inferring the exported model on the validation set.")
        predicted_validation_dataset = nncf_openvino_task.infer(
            validation_dataset.with_empty_annotations(),
            InferenceParameters(is_evaluation=True),
        )

        logger.info("Evaluating the exported model on the validation set.")
        resultset = ResultSetEntity(
            model=exported_model,
            ground_truth_dataset=validation_dataset,
            prediction_dataset=predicted_validation_dataset,
        )
        nncf_openvino_task.evaluate(resultset)
        logger.info(str(resultset.performance))


if __name__ == "__main__":
    sys.exit(main(parse_args()) or 0)
