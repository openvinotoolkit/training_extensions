# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import argparse
import sys

import numpy as np
from mmcv.utils import get_logger
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
from ote_sdk.usecases.tasks.interfaces.optimization_interface import OptimizationType
from torchreid_tasks.utils import get_task_class

import random
import torch
seed = 5
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logger = get_logger(name='sample')


def parse_args():
    parser = argparse.ArgumentParser(description='Sample showcasing the new API')
    parser.add_argument('template_file_path', help='path to template file')
    parser.add_argument('--export', action='store_true')
    return parser.parse_args()

def load_test_dataset(data_type):
    from ote_sdk.entities.annotation import Annotation, AnnotationSceneEntity, AnnotationSceneKind
    from ote_sdk.entities.dataset_item import DatasetItemEntity
    from ote_sdk.entities.image import Image
    from ote_sdk.entities.label import LabelEntity
    from ote_sdk.entities.scored_label import ScoredLabel
    from ote_sdk.entities.shapes.rectangle import Rectangle
    from ote_sdk.entities.subset import Subset
    import PIL

    def gen_image(resolution, shape=None):

        image = PIL.Image.new('RGB', resolution, (255, 255, 255))
        draw = PIL.ImageDraw.Draw(image)
        h, w = image.size
        if shape=='rectangle':
            draw.rectangle((h*0.25, w*0.25, h*0.75, w*0.75), fill=(0, 192, 192), outline=(0, 0, 0))
        if shape=='polygon':
            draw.polygon(((h*0.25, w*0.25), (h, w*0.25), (h*0.5,w)), fill=(255, 255, 0), outline=(0, 0, 0))
        return np.array(image)

    images = [
        gen_image((32, 32), shape='rectangle'),
        gen_image((32, 32), shape='polygon'),
    ]

    labels = [
        LabelEntity(name='rectangle', domain=Domain.CLASSIFICATION, id=0),
        LabelEntity(name='polygon', domain=Domain.CLASSIFICATION, id=1),  # NEW class
    ]

    def get_image(i, subset):
        return DatasetItemEntity(media=Image(data=images[i]),
                                 annotation_scene=AnnotationSceneEntity(
                                    annotations=[Annotation(
                                                 Rectangle(x1=0.0, y1=0.0, x2=1.0, y2=1.0),
                                                 labels=[ScoredLabel(label=labels[i])]
                                                 )],
                                    kind=AnnotationSceneKind.ANNOTATION
                                    ),
                                 subset=subset,
                                 )

    old_train = [
        get_image(0, Subset.TRAINING),
        get_image(0, Subset.TRAINING),
        get_image(0, Subset.TRAINING),
        get_image(0, Subset.TRAINING),
        get_image(0, Subset.TRAINING),
        get_image(0, Subset.TRAINING),
        get_image(0, Subset.TRAINING),
        get_image(0, Subset.TRAINING),
        get_image(0, Subset.TRAINING),
        get_image(0, Subset.TRAINING),
        get_image(0, Subset.TRAINING),
        get_image(0, Subset.TRAINING),
        get_image(0, Subset.TRAINING),
        get_image(0, Subset.TRAINING),
        get_image(0, Subset.TRAINING),
        get_image(0, Subset.TRAINING),
    ]

    old_val = [
        get_image(0, Subset.VALIDATION),
        get_image(0, Subset.VALIDATION),
        get_image(0, Subset.VALIDATION),
        get_image(0, Subset.VALIDATION),
        get_image(0, Subset.VALIDATION),
        get_image(0, Subset.VALIDATION),
        get_image(0, Subset.VALIDATION),
        get_image(0, Subset.VALIDATION),
        get_image(0, Subset.VALIDATION),
        get_image(0, Subset.VALIDATION),
        get_image(0, Subset.VALIDATION),
        get_image(0, Subset.VALIDATION),
        get_image(0, Subset.VALIDATION),
        get_image(0, Subset.VALIDATION),
        get_image(0, Subset.VALIDATION),
        get_image(0, Subset.VALIDATION),
    ]

    new_train = [
        get_image(0, Subset.TRAINING),
        get_image(1, Subset.TRAINING),
        get_image(0, Subset.TRAINING),
        get_image(1, Subset.TRAINING),
        get_image(0, Subset.TRAINING),
        get_image(1, Subset.TRAINING),
        get_image(0, Subset.TRAINING),
        get_image(1, Subset.TRAINING),
        get_image(0, Subset.TRAINING),
        get_image(1, Subset.TRAINING),
        get_image(0, Subset.TRAINING),
        get_image(1, Subset.TRAINING),
        get_image(0, Subset.TRAINING),
        get_image(1, Subset.TRAINING),
        get_image(0, Subset.TRAINING),
        get_image(1, Subset.TRAINING),
    ]

    new_val = [
        get_image(0, Subset.VALIDATION),
        get_image(1, Subset.VALIDATION),
        get_image(0, Subset.VALIDATION),
        get_image(1, Subset.VALIDATION),
        get_image(0, Subset.VALIDATION),
        get_image(1, Subset.VALIDATION),
        get_image(0, Subset.VALIDATION),
        get_image(1, Subset.VALIDATION),
        get_image(0, Subset.VALIDATION),
        get_image(1, Subset.VALIDATION),
        get_image(0, Subset.VALIDATION),
        get_image(1, Subset.VALIDATION),
        get_image(0, Subset.VALIDATION),
        get_image(1, Subset.VALIDATION),
        get_image(0, Subset.VALIDATION),
        get_image(1, Subset.VALIDATION),
    ]

    old = old_train + old_val
    new = new_train + new_val
    if data_type == 'old':
        return DatasetEntity(old), labels[:-1]
    else:
        return DatasetEntity(old + new), labels


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
    params.learning_parameters.learning_rate = 0.03
    params.learning_parameters.learning_rate_warmup_iters = 4
    params.learning_parameters.batch_size = 16

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
    params.learning_parameters.learning_rate = 0.03
    params.learning_parameters.learning_rate_warmup_iters = 4
    params.learning_parameters.batch_size = 16

    logger.info('Setup environment')
    environment = TaskEnvironment(
        model=initial_model,
        hyper_parameters=params,
        label_schema=labels_schema,
        model_template=model_template
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
