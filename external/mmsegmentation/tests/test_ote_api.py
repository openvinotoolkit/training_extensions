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

import os.path as osp
import random
import unittest
import warnings
from typing import Optional

import numpy as np
import pytest
from ote_sdk.test_suite.e2e_test_system import e2e_pytest_api
from ote_sdk.configuration.helper import create
from ote_sdk.entities.annotation import Annotation, AnnotationSceneEntity, AnnotationSceneKind
from ote_sdk.entities.dataset_item import DatasetItemEntity
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.image import Image
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.color import Color
from ote_sdk.entities.label import LabelEntity
from ote_sdk.entities.label_schema import LabelGroup, LabelGroupType, LabelSchemaEntity
from ote_sdk.entities.model import ModelEntity
from ote_sdk.entities.model_template import parse_model_template
from ote_sdk.entities.shapes.ellipse import Ellipse
from ote_sdk.entities.shapes.polygon import Polygon, Point
from ote_sdk.entities.shapes.rectangle import Rectangle
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.entities.train_parameters import TrainParameters
from ote_sdk.tests.test_helpers import generate_random_annotated_image

from segmentation_tasks.apis.segmentation import OTESegmentationTrainingTask


DEFAULT_TEMPLATE_DIR = osp.join('configs', 'custom-sematic-segmentation', 'ocr-lite-hrnet-18-mod2')


class API(unittest.TestCase):
    """
    Collection of tests for OTE API and OTE Model Templates
    """

    @staticmethod
    def generate_label_schema(label_names):
        label_domain = "segmentation"
        rgb = [int(i) for i in np.random.randint(0, 256, 3)]
        colors = [Color(*rgb) for _ in range(len(label_names))]
        not_empty_labels = [LabelEntity(name=name, color=colors[i], domain=label_domain, id=i) for i, name in
                            enumerate(label_names)]
        empty_label = LabelEntity(name=f"Empty label", color=Color(42, 43, 46),
                                  is_empty=True, domain=label_domain, id=len(not_empty_labels))

        label_schema = LabelSchemaEntity()
        exclusive_group = LabelGroup(name="labels", labels=not_empty_labels, group_type=LabelGroupType.EXCLUSIVE)
        empty_group = LabelGroup(name="empty", labels=[empty_label], group_type=LabelGroupType.EMPTY_LABEL)
        label_schema.add_group(exclusive_group)
        label_schema.add_group(empty_group)
        return label_schema

    def init_environment(self, params, model_template, number_of_images=10):
        labels_names = ('rectangle', 'ellipse', 'triangle')
        labels_schema = self.generate_label_schema(labels_names)
        labels_list = labels_schema.get_labels(False)
        environment = TaskEnvironment(model=None, hyper_parameters=params, label_schema=labels_schema,
                                      model_template=model_template)

        warnings.filterwarnings('ignore', message='.* coordinates .* are out of bounds.*')
        items = []
        for i in range(0, number_of_images):
            image_numpy, shapes = generate_random_annotated_image(image_width=640,
                                                                  image_height=480,
                                                                  labels=labels_list,
                                                                  max_shapes=20,
                                                                  min_size=50,
                                                                  max_size=100,
                                                                  random_seed=None)
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
            annotation = AnnotationSceneEntity(
                kind=AnnotationSceneKind.ANNOTATION,
                annotations=out_shapes)
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
        model_template = parse_model_template(osp.join(template_dir, 'template.yaml'))

        hyper_parameters = create(model_template.hyper_parameters.data)
        hyper_parameters.learning_parameters.learning_rate_fixed_iters = 0
        hyper_parameters.learning_parameters.learning_rate_warmup_iters = 0
        hyper_parameters.learning_parameters.num_iters = num_iters
        hyper_parameters.learning_parameters.num_checkpoints = 1

        return hyper_parameters, model_template

    @e2e_pytest_api
    def test_training_progress_tracking(self):
        hyper_parameters, model_template = self.setup_configurable_parameters(DEFAULT_TEMPLATE_DIR, num_iters=5)
        segmentation_environment, dataset = self.init_environment(hyper_parameters, model_template, 12)

        task = OTESegmentationTrainingTask(task_environment=segmentation_environment)
        self.addCleanup(task._delete_scratch_space)

        print('Task initialized, model training starts.')
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

        self.assertGreater(len(training_progress_curve), 0)
        training_progress_curve = np.asarray(training_progress_curve)
        self.assertTrue(np.all(training_progress_curve[1:] >= training_progress_curve[:-1]))

    @e2e_pytest_api
    def test_inference_progress_tracking(self):
        hyper_parameters, model_template = self.setup_configurable_parameters(DEFAULT_TEMPLATE_DIR, num_iters=10)
        segmentation_environment, dataset = self.init_environment(hyper_parameters, model_template, 12)

        task = OTESegmentationTrainingTask(task_environment=segmentation_environment)
        self.addCleanup(task._delete_scratch_space)

        print('Task initialized, model inference starts.')
        inference_progress_curve = []

        def progress_callback(progress: int):
            assert isinstance(progress, int)
            inference_progress_curve.append(progress)

        inference_parameters = InferenceParameters()
        inference_parameters.update_progress = progress_callback

        task.infer(dataset.with_empty_annotations(), inference_parameters)

        self.assertGreater(len(inference_progress_curve), 0)
        inference_progress_curve = np.asarray(inference_progress_curve)
        self.assertTrue(np.all(inference_progress_curve[1:] >= inference_progress_curve[:-1]))
        
    @e2e_pytest_api
    def test_nncf_optimize_progress_tracking(self):
        pytest.xfail('NNCF is not supported yet')
