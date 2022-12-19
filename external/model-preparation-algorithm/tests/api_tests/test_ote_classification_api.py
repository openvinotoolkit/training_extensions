# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os.path as osp
import random
from typing import Optional

import cv2 as cv
import numpy as np
import pytest
from bson import ObjectId
from mpa_tasks.apis.classification import (
    ClassificationInferenceTask,
    ClassificationTrainTask,
)
from ote_sdk.configuration.helper import create
from ote_sdk.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from ote_sdk.entities.dataset_item import DatasetItemEntity
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.id import ID
from ote_sdk.entities.image import Image
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.label import Domain, LabelEntity
from ote_sdk.entities.label_schema import LabelGroup, LabelGroupType, LabelSchemaEntity
from ote_sdk.entities.model import ModelEntity
from ote_sdk.entities.model_template import parse_model_template
from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.entities.shapes.rectangle import Rectangle
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.entities.train_parameters import TrainParameters
from ote_sdk.test_suite.e2e_test_system import e2e_pytest_api
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType

from tests.mpa_common import eval


DEFAULT_CLS_TEMPLATE_DIR = osp.join("configs", "classification", "efficientnet_b0_cls_incr")


class TestClassificationTaskAPI:
    @e2e_pytest_api
    def test_reading_classification_cls_incr_model_template(self):
        classification_template = [
            "efficientnet_b0_cls_incr",
            "efficientnet_v2_s_cls_incr",
            "mobilenet_v3_large_1_cls_incr",
        ]
        for model_template in classification_template:
            parse_model_template(osp.join("configs", "classification", model_template, "template.yaml"))

    @staticmethod
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

    @staticmethod
    def setup_configurable_parameters(template_dir, num_iters=10):
        model_template = parse_model_template(osp.join(template_dir, "template.yaml"))
        hyper_parameters = create(model_template.hyper_parameters.data)
        hyper_parameters.learning_parameters.num_iters = num_iters
        return hyper_parameters, model_template

    def init_environment(self, params, model_template, multilabel, hierarchical, number_of_images=10):
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
        labels_schema = self.generate_label_schema(
            dataset.get_labels(), multilabel=multilabel, hierarchical=hierarchical
        )
        environment = TaskEnvironment(
            model=None,
            hyper_parameters=params,
            label_schema=labels_schema,
            model_template=model_template,
        )
        return environment, dataset

    @e2e_pytest_api
    @pytest.mark.parametrize(
        "multilabel,hierarchical",
        [(False, False), (True, False), (False, True)],
        ids=["multiclass", "multilabel", "hierarchical"],
    )
    def test_training_progress_tracking(self, multilabel, hierarchical):
        hyper_parameters, model_template = self.setup_configurable_parameters(DEFAULT_CLS_TEMPLATE_DIR, num_iters=5)
        task_environment, dataset = self.init_environment(
            hyper_parameters, model_template, multilabel, hierarchical, 20
        )
        task = ClassificationTrainTask(task_environment=task_environment)
        print("Task initialized, model training starts.")

        training_progress_curve = []

        def progress_callback(progress: float, score: Optional[float] = None):
            training_progress_curve.append(progress)

        train_parameters = TrainParameters
        train_parameters.update_progress = progress_callback
        output_model = ModelEntity(
            dataset,
            task_environment.get_model_configuration(),
        )
        task.train(dataset, output_model, train_parameters)

        assert len(training_progress_curve) > 0
        assert np.all(training_progress_curve[1:] >= training_progress_curve[:-1])

    @e2e_pytest_api
    @pytest.mark.parametrize(
        "multilabel,hierarchical",
        [(False, False), (True, False), (False, True)],
        ids=["multiclass", "multilabel", "hierarchical"],
    )
    def test_inference_progress_tracking(self, multilabel, hierarchical):
        hyper_parameters, model_template = self.setup_configurable_parameters(DEFAULT_CLS_TEMPLATE_DIR, num_iters=5)
        task_environment, dataset = self.init_environment(
            hyper_parameters, model_template, multilabel, hierarchical, 20
        )
        task = ClassificationInferenceTask(task_environment=task_environment)
        print("Task initialized, model inference starts.")

        inference_progress_curve = []

        def progress_callback(progress: int):
            inference_progress_curve.append(progress)

        inference_parameters = InferenceParameters
        inference_parameters.update_progress = progress_callback
        task.infer(dataset.with_empty_annotations(), inference_parameters)

        assert len(inference_progress_curve) > 0
        assert np.all(inference_progress_curve[1:] >= inference_progress_curve[:-1])

    @e2e_pytest_api
    @pytest.mark.parametrize(
        "multilabel,hierarchical",
        [(False, False), (True, False), (False, True)],
        ids=["multiclass", "multilabel", "hierarchical"],
    )
    def test_inference_task(self, multilabel, hierarchical):
        # Prepare pretrained weights
        hyper_parameters, model_template = self.setup_configurable_parameters(DEFAULT_CLS_TEMPLATE_DIR, num_iters=2)
        classification_environment, dataset = self.init_environment(
            hyper_parameters, model_template, multilabel, hierarchical, 50
        )
        val_dataset = dataset.get_subset(Subset.VALIDATION)

        train_task = ClassificationTrainTask(task_environment=classification_environment)

        training_progress_curve = []

        def progress_callback(progress: float, score: Optional[float] = None):
            training_progress_curve.append(progress)

        train_parameters = TrainParameters
        train_parameters.update_progress = progress_callback
        trained_model = ModelEntity(
            dataset,
            classification_environment.get_model_configuration(),
        )
        train_task.train(dataset, trained_model, train_parameters)
        performance_after_train = eval(train_task, trained_model, val_dataset)

        # Create InferenceTask
        classification_environment.model = trained_model
        inference_task = ClassificationInferenceTask(task_environment=classification_environment)

        performance_after_load = eval(inference_task, trained_model, val_dataset)

        assert performance_after_train == performance_after_load

        # Export
        exported_model = ModelEntity(
            dataset,
            classification_environment.get_model_configuration(),
            _id=ObjectId(),
        )
        inference_task.export(ExportType.OPENVINO, exported_model)
