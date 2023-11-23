"""Sample code of otx training for classification."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=too-many-locals, invalid-name, too-many-statements

import argparse
import random
import sys

import numpy as np
import torch

from otx.algorithms.common.utils import get_task_class
from otx.api.configuration.helper import create
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.label import Domain
from otx.api.entities.label_schema import (
    LabelEntity,
    LabelGroup,
    LabelGroupType,
    LabelSchemaEntity,
)
from otx.api.entities.model import ModelEntity
from otx.api.entities.model_template import parse_model_template
from otx.api.entities.optimization_parameters import OptimizationParameters
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.subset import Subset
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from otx.api.usecases.tasks.interfaces.optimization_interface import OptimizationType
from otx.utils.logger import get_logger

SEED = 5
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logger = get_logger()


def parse_args():
    """Parse function for getting model template & check export."""
    parser = argparse.ArgumentParser(description="Sample showcasing the new API")
    parser.add_argument("template_file_path", help="path to template file")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--multilabel", action="store_true")
    parser.add_argument("--hierarchical", action="store_true")

    return parser.parse_args()


def load_test_dataset(data_type, args):
    """Load test dataset."""
    import PIL
    from PIL import ImageDraw

    from otx.api.entities.annotation import (
        Annotation,
        AnnotationSceneEntity,
        AnnotationSceneKind,
    )
    from otx.api.entities.dataset_item import DatasetItemEntity
    from otx.api.entities.image import Image
    from otx.api.entities.scored_label import ScoredLabel
    from otx.api.entities.shapes.rectangle import Rectangle

    def gen_image(resolution, shape=None):
        image = PIL.Image.new("RGB", resolution, (255, 255, 255))
        draw = ImageDraw.Draw(image)
        h, w = image.size
        shape = shape.split("+") if "+" in shape else [shape]
        for s in shape:
            if s == "rectangle":
                draw.rectangle((h * 0.1, w * 0.1, h * 0.4, w * 0.4), fill=(0, 192, 192), outline=(0, 0, 0))
            if s == "triangle":
                draw.polygon(
                    ((h * 0.5, w * 0.25), (h, w * 0.25), (h * 0.8, w * 0.5)), fill=(255, 255, 0), outline=(0, 0, 0)
                )
            if s == "pieslice":
                draw.pieslice(
                    ((h * 0.1, w * 0.5), (h * 0.5, w * 0.9)), start=50, end=250, fill=(0, 255, 0), outline=(0, 0, 0)
                )
            if s == "circle":
                draw.ellipse((h * 0.5, w * 0.5, h * 0.9, w * 0.9), fill="blue", outline="blue")
            if s == "text":
                draw.text((0, 0), "Intel", fill="blue", align="center")
        return np.array(image), shape

    datas = [
        gen_image((32, 32), shape="rectangle"),
        gen_image((32, 32), shape="triangle"),
        gen_image((32, 32), shape="rectangle+triangle"),  # for multilabel (old)
        gen_image((32, 32), shape="pieslice"),
        gen_image((32, 32), shape="pieslice+rectangle"),
        gen_image((32, 32), shape="pieslice+triangle"),
        gen_image((32, 32), shape="pieslice+rectangle+triangle"),  # for multilabel (new)
        gen_image((32, 32), shape="circle"),
        gen_image((32, 32), shape="circle+text"),  # for hierarchical (new)
    ]

    labels = {
        "rectangle": LabelEntity(name="rectangle", domain=Domain.CLASSIFICATION, id=0),
        "triangle": LabelEntity(name="triangle", domain=Domain.CLASSIFICATION, id=1),
        "pieslice": LabelEntity(name="pieslice", domain=Domain.CLASSIFICATION, id=2),
        "circle": LabelEntity(name="circle", domain=Domain.CLASSIFICATION, id=3),
        "text": LabelEntity(name="text", domain=Domain.CLASSIFICATION, id=4),
    }

    def get_image(i, subset, ignored_labels=None):
        image, shape = datas[i]
        lbl = [ScoredLabel(label=labels[s], probability=1.0) for s in shape]
        return DatasetItemEntity(
            media=Image(data=image),
            annotation_scene=AnnotationSceneEntity(
                annotations=[
                    Annotation(
                        Rectangle(x1=0.0, y1=0.0, x2=1.0, y2=1.0),
                        labels=lbl,
                    )
                ],
                kind=AnnotationSceneKind.ANNOTATION,
            ),
            subset=subset,
            ignored_labels=ignored_labels,
        )

    def gen_old_new_dataset(multilabel=False, hierarchical=False):
        old_train, old_val, new_train, new_val = [], [], [], []
        old_repeat = 8
        new_repeat = 4
        if multilabel:
            old_img_idx = [0, 1, 2]
            new_img_idx = [0, 1, 2, 3, 4, 5, 6]
            ignored_labels = [labels["pieslice"]]
        elif hierarchical:
            old_img_idx = [0, 1, 2, 3, 8]
            new_img_idx = [0, 1, 2, 3, 8, -1]
            ignored_labels = [labels["text"]]
        else:
            old_img_idx = [0, 1]
            new_img_idx = [0, 1, 3]

        for _ in range(old_repeat):
            for idx in old_img_idx:
                old_train.append(get_image(idx, Subset.TRAINING))
                old_val.append(get_image(idx, Subset.VALIDATION))
        for _ in range(new_repeat):
            for idx in new_img_idx:
                if multilabel:
                    new_train.append(get_image(idx, Subset.TRAINING, ignored_labels=ignored_labels))
                elif hierarchical:
                    new_train.append(get_image(idx, Subset.TRAINING, ignored_labels=ignored_labels))
                else:
                    new_train.append(get_image(idx, Subset.TRAINING))
                new_val.append(get_image(idx, Subset.VALIDATION))

        return old_train + old_val, new_train + new_val

    old, new = gen_old_new_dataset(args.multilabel, args.hierarchical)

    if not args.hierarchical:
        labels = [labels["rectangle"], labels["triangle"], labels["pieslice"]]
    else:
        labels = list(labels.values())

    if data_type == "old":
        return DatasetEntity(old), labels[:-1]
    return DatasetEntity(old + new), labels


def get_label_schema(labels, multilabel=False, hierarchical=False):
    """Get label schema."""
    label_schema = LabelSchemaEntity()

    emptylabel = LabelEntity(id=-1, name="Empty label", is_empty=True, domain=Domain.CLASSIFICATION)
    empty_group = LabelGroup(name="empty", labels=[emptylabel], group_type=LabelGroupType.EMPTY_LABEL)

    if multilabel:
        for label in labels:
            label_schema.add_group(LabelGroup(name=label.name, labels=[label], group_type=LabelGroupType.EXCLUSIVE))
        label_schema.add_group(empty_group)
    elif hierarchical:
        single_label_classes = ["pieslice", "circle"]
        multi_label_classes = ["rectangle", "triangle", "text"]

        single_labels = [label for label in labels if label.name in single_label_classes]
        single_label_group = LabelGroup(name="labels", labels=single_labels, group_type=LabelGroupType.EXCLUSIVE)
        label_schema.add_group(single_label_group)

        for label in labels:
            if label.name in multi_label_classes:
                label_schema.add_group(
                    LabelGroup(
                        name=f"{label.name}____{label.name}_group", labels=[label], group_type=LabelGroupType.EXCLUSIVE
                    )
                )
    else:
        main_group = LabelGroup(name="labels", labels=labels, group_type=LabelGroupType.EXCLUSIVE)
        label_schema.add_group(main_group)

    return label_schema


def validate(task, validation_dataset, model):
    """Validate."""
    print("Get predictions on the validation set")
    predicted_validation_dataset = task.infer(
        validation_dataset.with_empty_annotations(), InferenceParameters(is_evaluation=True)
    )
    resultset = ResultSetEntity(
        model=model,
        ground_truth_dataset=validation_dataset,
        prediction_dataset=predicted_validation_dataset,
    )
    print("Estimate quality on validation set")
    task.evaluate(resultset)
    print(str(resultset.performance))


def main(args):
    """Main of Classification Sample Test."""

    logger.info("Train initial model with OLD dataset")
    dataset, labels_list = load_test_dataset("old", args)
    labels_schema = get_label_schema(labels_list, multilabel=args.multilabel, hierarchical=args.hierarchical)

    logger.info(f"Train dataset: {len(dataset.get_subset(Subset.TRAINING))} items")
    logger.info(f"Validation dataset: {len(dataset.get_subset(Subset.VALIDATION))} items")

    logger.info("Load model template")
    model_template = parse_model_template(args.template_file_path)

    logger.info("Set hyperparameters")
    params = create(model_template.hyper_parameters.data)
    params.learning_parameters.num_iters = 10
    params.learning_parameters.learning_rate = 0.03
    params.learning_parameters.learning_rate_warmup_iters = 4
    params.learning_parameters.batch_size = 16

    logger.info("Setup environment")
    environment = TaskEnvironment(
        model=None, hyper_parameters=params, label_schema=labels_schema, model_template=model_template
    )

    logger.info("Create base Task")
    task_impl_path = model_template.entrypoints.base
    task_cls = get_task_class(task_impl_path)
    task = task_cls(task_environment=environment)

    logger.info("Train model")
    initial_model = ModelEntity(
        dataset,
        environment.get_model_configuration(),
    )
    task.train(dataset, initial_model)

    logger.info("Class-incremental learning with OLD + NEW dataset")
    dataset, labels_list = load_test_dataset("new")
    labels_schema = get_label_schema(labels_list, multilabel=args.multilabel, hierarchical=args.hierarchical)

    logger.info(f"Train dataset: {len(dataset.get_subset(Subset.TRAINING))} items")
    logger.info(f"Validation dataset: {len(dataset.get_subset(Subset.VALIDATION))} items")

    logger.info("Load model template")
    model_template = parse_model_template(args.template_file_path)

    logger.info("Set hyperparameters")
    params = create(model_template.hyper_parameters.data)
    params.learning_parameters.num_iters = 10
    params.learning_parameters.learning_rate = 0.03
    params.learning_parameters.learning_rate_warmup_iters = 4
    params.learning_parameters.batch_size = 16

    logger.info("Setup environment")
    environment = TaskEnvironment(
        model=initial_model, hyper_parameters=params, label_schema=labels_schema, model_template=model_template
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
        validation_dataset.with_empty_annotations(), InferenceParameters(is_evaluation=True)
    )
    resultset = ResultSetEntity(
        model=output_model,
        ground_truth_dataset=validation_dataset,
        prediction_dataset=predicted_validation_dataset,
    )
    logger.info("Estimate quality on validation set")
    task.evaluate(resultset)
    logger.info(str(resultset.performance))

    if args.export or args.multilabel or args.hierarchical:
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
            validation_dataset.with_empty_annotations(), InferenceParameters(is_evaluation=True)
        )

        resultset = ResultSetEntity(
            model=exported_model,
            ground_truth_dataset=validation_dataset,
            prediction_dataset=predicted_validation_dataset,
        )
        logger.info("Estimate quality on validation set")
        openvino_task.evaluate(resultset)
        logger.info(str(resultset.performance))

        # POT test
        logger.info("Run POT optimization")
        optimized_model = ModelEntity(
            dataset,
            environment.get_model_configuration(),
        )
        openvino_task.optimize(OptimizationType.POT, dataset, optimized_model, OptimizationParameters())
        logger.info("Run POT deploy")
        openvino_task.deploy(optimized_model)
        validate(task, validation_dataset, optimized_model)

        # NNCF test
        task_impl_path = model_template.entrypoints.nncf
        nncf_task_cls = get_task_class(task_impl_path)

        print("Create NNCF Task")
        environment.model = output_model
        task = nncf_task_cls(task_environment=environment)

        print("Optimize model by NNCF")
        optimized_model = ModelEntity(
            dataset,
            environment.get_model_configuration(),
        )
        task.optimize(OptimizationType.NNCF, dataset, optimized_model, None)
        validate(task, validation_dataset, output_model)


if __name__ == "__main__":
    sys.exit(main(parse_args()) or 0)
