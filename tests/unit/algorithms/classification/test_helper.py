# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import random
from pathlib import Path

import cv2 as cv
import numpy as np

from otx.algorithms.common.adapters.mmcv.utils.config_utils import MPAConfig
from otx.api.configuration.helper import create
from otx.api.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.id import ID
from otx.api.entities.image import Image
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.label_schema import LabelGroup, LabelGroupType, LabelSchemaEntity
from otx.api.entities.model_template import parse_model_template
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.subset import Subset
from otx.api.entities.task_environment import TaskEnvironment

DEFAULT_CLS_TEMPLATE_DIR = (
    Path("src") / "otx" / "algorithms" / "classification" / "configs" / "mobilenet_v3_large_1_cls_incr"
)
DEFAULT_CLS_TEMPLATE = DEFAULT_CLS_TEMPLATE_DIR / "template.yaml"


def generate_label_schema(not_empty_labels, multilabel=False, hierarchical=False):
    assert len(not_empty_labels) > 1

    label_schema = LabelSchemaEntity()
    if multilabel:
        emptylabel = LabelEntity(name="Empty label", is_empty=True, domain=Domain.CLASSIFICATION)
        empty_group = LabelGroup(name="empty", labels=[emptylabel], group_type=LabelGroupType.EMPTY_LABEL)
        for label in not_empty_labels:
            label_schema.add_group(
                LabelGroup(
                    name=label.name,
                    labels=[label],
                    group_type=LabelGroupType.EXCLUSIVE,
                )
            )
        label_schema.add_group(empty_group)
    elif hierarchical:
        single_label_classes = ["b", "g", "r"]
        multi_label_classes = ["w", "p"]
        emptylabel = LabelEntity(name="Empty label", is_empty=True, domain=Domain.CLASSIFICATION)
        empty_group = LabelGroup(name="empty", labels=[emptylabel], group_type=LabelGroupType.EMPTY_LABEL)
        single_labels = []
        for label in not_empty_labels:
            if label.name in multi_label_classes:
                label_schema.add_group(
                    LabelGroup(
                        name=label.name,
                        labels=[label],
                        group_type=LabelGroupType.EXCLUSIVE,
                    )
                )
                if empty_group not in label_schema.get_groups(include_empty=True):
                    label_schema.add_group(empty_group)
            elif label.name in single_label_classes:
                single_labels.append(label)
        if single_labels:
            single_label_group = LabelGroup(
                name="labels",
                labels=single_labels,
                group_type=LabelGroupType.EXCLUSIVE,
            )
            label_schema.add_group(single_label_group)
    else:
        main_group = LabelGroup(
            name="labels",
            labels=not_empty_labels,
            group_type=LabelGroupType.EXCLUSIVE,
        )
        label_schema.add_group(main_group)
    return label_schema


def generate_cls_dataset(hierarchical=False, number_of_images=10):
    resolution = (224, 224)
    if hierarchical:
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 0, 0), (230, 230, 250)]
        cls_names = ["b", "g", "r", "w", "p"]
        texts = ["Blue", "Green", "Red", "White", "Purple"]
    else:
        colors = [(0, 255, 0), (0, 0, 255)]
        cls_names = ["b", "g"]
        texts = ["Blue", "Green"]
    env_labels = [
        LabelEntity(name=name, domain=Domain.CLASSIFICATION, is_empty=False, id=ID(i))
        for i, name in enumerate(cls_names)
    ]

    items = []

    for _ in range(0, number_of_images):
        for j, lbl in enumerate(env_labels):
            class_img = np.zeros((*resolution, 3), dtype=np.uint8)
            class_img[:] = colors[j]
            class_img = cv.putText(
                class_img,
                texts[j],
                (50, 50),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8 + j * 0.2,
                colors[j - 1],
                2,
                cv.LINE_AA,
            )

            image = Image(data=class_img)
            labels = [ScoredLabel(label=lbl, probability=1.0)]
            shapes = [Annotation(Rectangle.generate_full_box(), labels)]
            annotation_scene = AnnotationSceneEntity(kind=AnnotationSceneKind.ANNOTATION, annotations=shapes)
            items.append(DatasetItemEntity(media=image, annotation_scene=annotation_scene))

    rng = random.Random()
    rng.seed(100)
    rng.shuffle(items)
    for i, _ in enumerate(items):
        subset_region = i / number_of_images
        if subset_region >= 0.9:
            subset = Subset.TESTING
        elif subset_region >= 0.6:
            subset = Subset.VALIDATION
        else:
            subset = Subset.TRAINING
        items[i].subset = subset

    dataset = DatasetEntity(items)
    return dataset


def init_environment(params, model_template, multilabel=False, hierarchical=False, number_of_images=10):
    dataset = generate_cls_dataset(hierarchical, number_of_images)
    labels_schema = generate_label_schema(dataset.get_labels(), multilabel=multilabel, hierarchical=hierarchical)
    environment = TaskEnvironment(
        model=None,
        hyper_parameters=params,
        label_schema=labels_schema,
        model_template=model_template,
    )
    return environment, dataset


def setup_configurable_parameters(template_dir, num_iters=10):
    model_template = parse_model_template(str(template_dir))
    hyper_parameters = create(model_template.hyper_parameters.data)
    hyper_parameters.learning_parameters.num_iters = num_iters

    return hyper_parameters, model_template


def setup_mpa_task_parameters(task_type, create_val=False, create_test=False):
    if task_type == "semisl":
        recipie_path = "src/otx/recipes/stages/classification/semisl.yaml"
    elif task_type == "incremental":
        recipie_path = "src/otx/recipes/stages/classification/incremental.yaml"
    recipie_cfg = MPAConfig.fromfile(recipie_path)
    model_cfg = MPAConfig.fromfile(DEFAULT_CLS_TEMPLATE_DIR / "model.py")
    model_cfg.model.multilabel = False
    model_cfg.model.hierarchical = False
    data_cfg = MPAConfig.fromfile(DEFAULT_CLS_TEMPLATE_DIR / "data_pipeline.py")
    data_cfg.data.train.data_dir = "tests/assets/classification_dataset"
    if create_val:
        data_cfg.data.val.data_dir = "tests/assets/classification_dataset"
    else:
        data_cfg.data.val = None
    if create_test:
        data_cfg.data.test.data_dir = "tests/assets/classification_dataset"
    else:
        data_cfg.data.test = None
    dummy_dataset = generate_cls_dataset(number_of_images=1)
    data_cfg.data.train.otx_dataset = dummy_dataset
    data_cfg.data.train.labels = dummy_dataset.get_labels()
    data_cfg.data.train.data_classes = ["label_0", "label_1"]
    data_cfg.data.train.new_classes = ["label_0", "label_1", "label_3"]

    return model_cfg, data_cfg, recipie_cfg
