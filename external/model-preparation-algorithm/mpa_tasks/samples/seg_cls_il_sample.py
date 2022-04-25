# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import argparse
import sys

import cv2
import numpy as np
from mmcv.utils import get_logger
from segmentation_tasks.apis.segmentation.ote_utils import get_task_class
from ote_sdk.configuration.helper import create
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.label import Domain
from ote_sdk.entities.label_schema import LabelSchemaEntity
from ote_sdk.entities.model import ModelEntity
from ote_sdk.entities.model_template import parse_model_template
from ote_sdk.entities.optimization_parameters import OptimizationParameters
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType
from ote_sdk.usecases.tasks.interfaces.optimization_interface import \
    OptimizationType

logger = get_logger(name='sample')


def parse_args():
    parser = argparse.ArgumentParser(description='Sample showcasing the new API')
    parser.add_argument('template_file_path', help='path to template file')
    parser.add_argument('--export', action='store_true')
    return parser.parse_args()


colors = [(128, 0, 0), (0, 128, 0)]


def load_test_dataset(data_type):
    from ote_sdk.entities.annotation import (Annotation, AnnotationSceneEntity,
                                             AnnotationSceneKind)
    from ote_sdk.entities.dataset_item import DatasetItemEntity
    from ote_sdk.entities.image import Image
    from ote_sdk.entities.label import LabelEntity
    from ote_sdk.entities.scored_label import ScoredLabel
    from ote_sdk.entities.shapes.polygon import Point, Polygon
    from ote_sdk.entities.subset import Subset

    def gen_circle_image(resolution):
        w, h = resolution
        image = np.full([h, w, 3], fill_value=128, dtype=np.uint8)
        gt = np.full([h, w, 1], fill_value=0, dtype=np.uint8)
        cv2.circle(image, (int(h/2), int(w/2)), 90, (0, 0, 255), -1)
        cv2.circle(gt, (int(h/2), int(w/2)), 90, 1, -1)
        return (image, gt)

    def gen_rect_image(resolution):
        w, h = resolution
        image = np.full([h, w, 3], fill_value=128, dtype=np.uint8)
        gt = np.full([h, w, 1], fill_value=0, dtype=np.uint8)
        cv2.rectangle(image, (int(h*0.1), int(w*0.1)), (int(h/2), int(w/2)), (0, 255, 0), -1)
        cv2.rectangle(gt, (int(h*0.1), int(w*0.1)), (int(h/2), int(w/2)), 2, -1)
        return (image, gt)

    labels = [
        LabelEntity(name='circle', domain=Domain.SEGMENTATION, id=1),  # OLD class
        LabelEntity(name='rect', domain=Domain.SEGMENTATION, id=2),
    ]

    def get_image(subset, label_id):
        if label_id == 1:
            image, gt = gen_circle_image((640, 480))
        else:
            image, gt = gen_rect_image((640, 480))

        height, width = gt.shape[:2]
        label_mask = (gt == label_id)
        label_index_map = label_mask.astype(np.uint8)
        contours, hierarchies = cv2.findContours(
            label_index_map, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour, hierarchy in zip(contours, hierarchies[0]):
            if hierarchy[3] != -1:
                continue

            contour = list(contour)
            if len(contour) <= 2:
                continue

            points = [
                Point(x=point[0][0] / width, y=point[0][1] / height)
                for point in contour
            ]
        return DatasetItemEntity(
            media=Image(data=image),
            annotation_scene=AnnotationSceneEntity(
                annotations=[Annotation(Polygon(points=points), labels=[ScoredLabel(label=labels[label_id-1])])],
                kind=AnnotationSceneKind.ANNOTATION
            ),
            subset=subset,
        )

    old_train = [
        get_image(Subset.TRAINING, 1),
        get_image(Subset.TRAINING, 1),
        get_image(Subset.TRAINING, 1),
        get_image(Subset.TRAINING, 1),
        get_image(Subset.TRAINING, 1),
        get_image(Subset.TRAINING, 1),
        get_image(Subset.TRAINING, 1),
        get_image(Subset.TRAINING, 1),
    ]

    old_val = [
        get_image(Subset.VALIDATION, 1),
        get_image(Subset.VALIDATION, 1),
        get_image(Subset.VALIDATION, 1),
        get_image(Subset.VALIDATION, 1),
        get_image(Subset.VALIDATION, 1),
        get_image(Subset.VALIDATION, 1),
        get_image(Subset.VALIDATION, 1),
        get_image(Subset.VALIDATION, 1),
    ]

    new_train = [
        get_image(Subset.TRAINING, 1),
        get_image(Subset.TRAINING, 1),
        get_image(Subset.TRAINING, 1),
        get_image(Subset.TRAINING, 1),
        get_image(Subset.TRAINING, 2),
        get_image(Subset.TRAINING, 2),
        get_image(Subset.TRAINING, 2),
        get_image(Subset.TRAINING, 2),
    ]

    new_val = [
        get_image(Subset.VALIDATION, 1),
        get_image(Subset.VALIDATION, 1),
        get_image(Subset.VALIDATION, 1),
        get_image(Subset.VALIDATION, 1),
        get_image(Subset.VALIDATION, 2),
        get_image(Subset.VALIDATION, 2),
        get_image(Subset.VALIDATION, 2),
        get_image(Subset.VALIDATION, 2),
    ]
    old = old_train + old_val
    new = new_train + new_val
    if data_type == 'old':
        return DatasetEntity(old*10), [labels[0]]
    else:
        return DatasetEntity((old*10 + new*10)), labels


def main(args):
    logger.info('Train initial model with OLD dataset')
    dataset, labels_list = load_test_dataset('old')
    labels_schema = LabelSchemaEntity.from_labels(labels_list)

    logger.info(f'Train dataset: {len(dataset.get_subset(Subset.TRAINING))} items')
    logger.info(f'Validation dataset: {len(dataset.get_subset(Subset.VALIDATION))} items')

    logger.info('Load model template')
    model_template = parse_model_template(args.template_file_path)

    logger.info('Set hyperparameters')
    params = create(model_template.hyper_parameters.data)
    params.learning_parameters.num_iters = 10
    params.learning_parameters.batch_size = 8

    logger.info('Setup environment')
    environment = TaskEnvironment(
        model=None,
        hyper_parameters=params,
        label_schema=labels_schema,
        model_template=model_template
    )

    logger.info('Create base Task')
    task_impl_path = model_template.entrypoints.base
    task_cls = get_task_class(task_impl_path)
    task = task_cls(task_environment=environment)
    task.freeze = False

    logger.info('Train model')
    initial_model = ModelEntity(
        dataset,
        environment.get_model_configuration(),
    )
    task.train(dataset, initial_model)

    logger.info('Class-incremental learning with OLD + NEW dataset')
    dataset, labels_list = load_test_dataset('new')
    labels_schema = LabelSchemaEntity.from_labels(labels_list)

    logger.info(f'Train dataset: {len(dataset.get_subset(Subset.TRAINING))} items')
    logger.info(f'Validation dataset: {len(dataset.get_subset(Subset.VALIDATION))} items')

    logger.info('Load model template')
    model_template = parse_model_template(args.template_file_path)

    logger.info('Set hyperparameters')
    params = create(model_template.hyper_parameters.data)
    params.learning_parameters.num_iters = 10
    params.learning_parameters.batch_size = 8

    logger.info('Setup environment')
    environment = TaskEnvironment(model=initial_model, hyper_parameters=params,
                                  label_schema=labels_schema, model_template=model_template)

    environment.model = ModelEntity(
        train_dataset=dataset,
        configuration=environment.get_model_configuration(),
        model_adapters=initial_model.model_adapters
    )

    logger.info('Create base Task')
    task_impl_path = model_template.entrypoints.base
    task_cls = get_task_class(task_impl_path)
    task = task_cls(task_environment=environment)

    logger.info('Train model')
    output_model = ModelEntity(
        dataset,
        environment.get_model_configuration(),
    )
    task.train(dataset, output_model)

    logger.info('Get predictions on the validation set')
    validation_dataset = dataset.get_subset(Subset.VALIDATION)
    predicted_validation_dataset = task.infer(
        validation_dataset.with_empty_annotations(),
        InferenceParameters(is_evaluation=True))
    resultset = ResultSetEntity(
        model=output_model,
        ground_truth_dataset=validation_dataset,
        prediction_dataset=predicted_validation_dataset,
    )
    logger.info('Estimate quality on validation set')
    task.evaluate(resultset)
    logger.info(str(resultset.performance))

    if args.export:
        logger.info('Export model')
        exported_model = ModelEntity(
            dataset,
            environment.get_model_configuration(),
        )
        task.export(ExportType.OPENVINO, exported_model)

        logger.info('Create OpenVINO Task')
        environment.model = exported_model
        openvino_task_impl_path = model_template.entrypoints.openvino
        openvino_task_cls = get_task_class(openvino_task_impl_path)
        openvino_task = openvino_task_cls(environment)

        logger.info('Get predictions on the validation set')
        predicted_validation_dataset = openvino_task.infer(
            validation_dataset.with_empty_annotations(),
            InferenceParameters(is_evaluation=True))
        resultset = ResultSetEntity(
            model=output_model,
            ground_truth_dataset=validation_dataset,
            prediction_dataset=predicted_validation_dataset,
        )
        logger.info('Estimate quality on validation set')
        openvino_task.evaluate(resultset)
        logger.info(str(resultset.performance))

        logger.info('Run POT optimization')
        optimized_model = ModelEntity(
            dataset,
            environment.get_model_configuration(),
        )
        openvino_task.optimize(
            OptimizationType.POT,
            dataset.get_subset(Subset.TRAINING),
            optimized_model,
            OptimizationParameters())

        logger.info('Get predictions on the validation set')
        predicted_validation_dataset = openvino_task.infer(
            validation_dataset.with_empty_annotations(),
            InferenceParameters(is_evaluation=True))
        resultset = ResultSetEntity(
            model=optimized_model,
            ground_truth_dataset=validation_dataset,
            prediction_dataset=predicted_validation_dataset,
        )
        logger.info('Performance of optimized model:')
        openvino_task.evaluate(resultset)
        logger.info(str(resultset.performance))


if __name__ == '__main__':
    sys.exit(main(parse_args()) or 0)
