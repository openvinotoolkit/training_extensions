"""API Tests for segmentation training"""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os.path as osp
import random
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
from bson import ObjectId

from otx.algorithms.common.tasks.training_base import BaseTask
from otx.algorithms.segmentation.tasks import (
    SegmentationInferenceTask,
    SegmentationTrainTask,
)
from otx.api.configuration.helper import create
from otx.api.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from otx.api.entities.color import Color
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.image import Image
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.label_schema import LabelGroup, LabelGroupType, LabelSchemaEntity
from otx.api.entities.metrics import Performance
from otx.api.entities.model import ModelEntity
from otx.api.entities.model_template import parse_model_template
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.shapes.ellipse import Ellipse
from otx.api.entities.shapes.polygon import Point, Polygon
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.subset import Subset
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.entities.train_parameters import TrainParameters
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from tests.test_helpers import generate_random_annotated_image
from tests.test_suite.e2e_test_system import e2e_pytest_api

DEFAULT_SEG_TEMPLATE_DIR = osp.join("src/otx/algorithms/segmentation/configs", "ocr_lite_hrnet_18_mod2")


def task_eval(task: BaseTask, model: ModelEntity, dataset: DatasetEntity) -> Performance:
    start_time = time.time()
    result_dataset = task.infer(dataset.with_empty_annotations())
    end_time = time.time()
    print(f"{len(dataset)} analysed in {end_time - start_time} seconds")
    result_set = ResultSetEntity(model=model, ground_truth_dataset=dataset, prediction_dataset=result_dataset)
    task.evaluate(result_set)
    assert result_set.performance is not None
    return result_set.performance


class TestOTXSegAPI:
    """
    Collection of tests for OTX API and OTX Model Templates
    """

    @e2e_pytest_api
    def test_reading_segmentation_cls_incr_model_template(self):
        segmentation_template = [
            "ocr_lite_hrnet_18_mod2",
            "ocr_lite_hrnet_s_mod2",
            "ocr_lite_hrnet_x_mod3",
        ]
        for model_template in segmentation_template:
            parse_model_template(osp.join("src/otx/algorithms/segmentation/configs", model_template, "template.yaml"))

    @staticmethod
    def generate_label_schema(label_names):
        label_domain = Domain.SEGMENTATION
        rgb = [int(i) for i in np.random.randint(0, 256, 3)]
        colors = [Color(*rgb) for _ in range(len(label_names))]
        not_empty_labels = [
            LabelEntity(name=name, color=colors[i], domain=label_domain, id=i) for i, name in enumerate(label_names)
        ]
        empty_label = LabelEntity(
            name="Empty label",
            color=Color(42, 43, 46),
            is_empty=True,
            domain=label_domain,
            id=len(not_empty_labels),
        )

        label_schema = LabelSchemaEntity()
        exclusive_group = LabelGroup(name="labels", labels=not_empty_labels, group_type=LabelGroupType.EXCLUSIVE)
        empty_group = LabelGroup(name="empty", labels=[empty_label], group_type=LabelGroupType.EMPTY_LABEL)
        label_schema.add_group(exclusive_group)
        label_schema.add_group(empty_group)
        return label_schema

    def init_environment(self, params, model_template, number_of_images=10):
        labels_names = ("rectangle", "ellipse", "triangle")
        labels_schema = self.generate_label_schema(labels_names)
        labels_list = labels_schema.get_labels(False)
        environment = TaskEnvironment(
            model=None,
            hyper_parameters=params,
            label_schema=labels_schema,
            model_template=model_template,
        )

        warnings.filterwarnings("ignore", message=".* coordinates .* are out of bounds.*")
        items = []
        for i in range(0, number_of_images):
            image_numpy, shapes = generate_random_annotated_image(
                image_width=640,
                image_height=480,
                labels=labels_list,
                max_shapes=20,
                min_size=50,
                max_size=100,
                random_seed=None,
            )
            # Convert all shapes to polygons
            out_shapes = []
            for shape in shapes:
                shape_labels = shape.get_labels(include_empty=True)

                in_shape = shape.shape
                if isinstance(in_shape, Rectangle):
                    points = [
                        Point(in_shape.x1, in_shape.y1),
                        Point(in_shape.x2, in_shape.y1),
                        Point(in_shape.x2, in_shape.y2),
                        Point(in_shape.x1, in_shape.y2),
                    ]
                elif isinstance(in_shape, Ellipse):
                    points = [Point(x, y) for x, y in in_shape.get_evenly_distributed_ellipse_coordinates()]
                elif isinstance(in_shape, Polygon):
                    points = in_shape.points

                out_shapes.append(Annotation(Polygon(points=points), labels=shape_labels))

            image = Image(data=image_numpy)
            annotation = AnnotationSceneEntity(kind=AnnotationSceneKind.ANNOTATION, annotations=out_shapes)
            items.append(DatasetItemEntity(media=image, annotation_scene=annotation))
        warnings.resetwarnings()

        rng = random.Random()
        rng.shuffle(items)
        for i, _ in enumerate(items):
            subset_region = i / number_of_images
            if subset_region >= 0.8:
                subset = Subset.TESTING
            elif subset_region >= 0.6:
                subset = Subset.VALIDATION
            else:
                subset = Subset.TRAINING

            items[i].subset = subset

        dataset = DatasetEntity(items)

        return environment, dataset

    @staticmethod
    def setup_configurable_parameters(template_dir, num_iters=10):
        model_template = parse_model_template(osp.join(template_dir, "template.yaml"))

        hyper_parameters = create(model_template.hyper_parameters.data)
        hyper_parameters.learning_parameters.learning_rate_warmup_iters = 1
        hyper_parameters.learning_parameters.num_iters = num_iters
        hyper_parameters.learning_parameters.num_checkpoints = 1

        return hyper_parameters, model_template

    @e2e_pytest_api
    def test_cancel_training_segmentation(self):
        """
        Tests starting and cancelling training.

        Flow of the test:
        - Creates a randomly annotated project with a small dataset.
        - Start training and give cancel training signal after 10 seconds. Assert that training
            stops within 35 seconds after that
        - Start training and give cancel signal immediately. Assert that training stops within 25 seconds.

        This test should be finished in under one minute on a workstation.
        """
        hyper_parameters, model_template = self.setup_configurable_parameters(DEFAULT_SEG_TEMPLATE_DIR, num_iters=200)
        segmentation_environment, dataset = self.init_environment(hyper_parameters, model_template, 64)

        segmentation_task = SegmentationTrainTask(task_environment=segmentation_environment)

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="train_thread")

        output_model = ModelEntity(
            dataset,
            segmentation_environment.get_model_configuration(),
        )

        training_progress_curve = []

        def progress_callback(progress: float, score: Optional[float] = None):
            training_progress_curve.append(progress)

        train_parameters = TrainParameters()
        train_parameters.update_progress = progress_callback

        # Test stopping after some time
        start_time = time.time()
        train_future = executor.submit(segmentation_task.train, dataset, output_model, train_parameters)
        # give train_thread some time to initialize the model
        while not segmentation_task._is_training:
            time.sleep(10)
        segmentation_task.cancel_training()

        # stopping process has to happen in less than 35 seconds
        train_future.result()
        assert training_progress_curve[-1] == 100
        assert time.time() - start_time < 100, "Expected to stop within 100 seconds."

        # Test stopping immediately
        start_time = time.time()
        train_future = executor.submit(segmentation_task.train, dataset, output_model)
        segmentation_task.cancel_training()

        train_future.result()
        assert time.time() - start_time < 25  # stopping process has to happen in less than 25 seconds

    @e2e_pytest_api
    def test_training_progress_tracking(self):
        hyper_parameters, model_template = self.setup_configurable_parameters(DEFAULT_SEG_TEMPLATE_DIR, num_iters=5)
        segmentation_environment, dataset = self.init_environment(hyper_parameters, model_template, 12)

        task = SegmentationTrainTask(task_environment=segmentation_environment)
        print("Task initialized, model training starts.")

        training_progress_curve = []

        def progress_callback(progress: float, score: Optional[float] = None):
            training_progress_curve.append(progress)

        train_parameters = TrainParameters()
        train_parameters.update_progress = progress_callback
        output_model = ModelEntity(
            dataset,
            segmentation_environment.get_model_configuration(),
        )
        task.train(dataset, output_model, train_parameters)

        assert len(training_progress_curve) > 0
        assert np.all(training_progress_curve[1:] >= training_progress_curve[:-1])

    @e2e_pytest_api
    def test_inference_progress_tracking(self):
        hyper_parameters, model_template = self.setup_configurable_parameters(DEFAULT_SEG_TEMPLATE_DIR, num_iters=10)
        segmentation_environment, dataset = self.init_environment(hyper_parameters, model_template, 12)

        task = SegmentationInferenceTask(task_environment=segmentation_environment)
        print("Task initialized, model inference starts.")

        inference_progress_curve = []

        def progress_callback(progress: int):
            assert isinstance(progress, int)
            inference_progress_curve.append(progress)

        inference_parameters = InferenceParameters()
        inference_parameters.update_progress = progress_callback
        task.infer(dataset.with_empty_annotations(), inference_parameters)

        assert len(inference_progress_curve) > 0
        assert np.all(inference_progress_curve[1:] >= inference_progress_curve[:-1])

    @e2e_pytest_api
    def test_inference_task(self):
        # Prepare pretrained weights
        hyper_parameters, model_template = self.setup_configurable_parameters(DEFAULT_SEG_TEMPLATE_DIR, num_iters=2)
        segmentation_environment, dataset = self.init_environment(hyper_parameters, model_template, 30)
        val_dataset = dataset.get_subset(Subset.VALIDATION)

        train_task = SegmentationTrainTask(task_environment=segmentation_environment)

        training_progress_curve = []

        def progress_callback(progress: float, score: Optional[float] = None):
            training_progress_curve.append(progress)

        train_parameters = TrainParameters()
        train_parameters.update_progress = progress_callback
        trained_model = ModelEntity(
            dataset,
            segmentation_environment.get_model_configuration(),
        )
        train_task.train(dataset, trained_model, train_parameters)
        performance_after_train = task_eval(train_task, trained_model, val_dataset)

        # Create InferenceTask
        segmentation_environment.model = trained_model
        inference_task = SegmentationInferenceTask(task_environment=segmentation_environment)

        performance_after_load = task_eval(inference_task, trained_model, val_dataset)

        assert performance_after_train == performance_after_load

        # Export
        exported_model = ModelEntity(dataset, segmentation_environment.get_model_configuration(), _id=ObjectId())
        inference_task.export(ExportType.OPENVINO, exported_model)
