# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import glob
import os.path as osp
import random
import time
import unittest
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import cv2 as cv
import numpy as np
from bson import ObjectId
from detection_tasks.apis.detection.ote_utils import generate_label_schema
from mpa_tasks.apis import BaseTask
from mpa_tasks.apis.classification import (
    ClassificationInferenceTask,
    ClassificationTrainTask,
)
from mpa_tasks.apis.detection import DetectionInferenceTask, DetectionTrainTask
from mpa_tasks.apis.segmentation import SegmentationInferenceTask, SegmentationTrainTask
from ote_sdk.configuration.helper import create
from ote_sdk.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from ote_sdk.entities.color import Color
from ote_sdk.entities.dataset_item import DatasetItemEntity
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.id import ID
from ote_sdk.entities.image import Image
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.label import Domain, LabelEntity
from ote_sdk.entities.label_schema import LabelGroup, LabelGroupType, LabelSchemaEntity
from ote_sdk.entities.metrics import Performance
from ote_sdk.entities.model import ModelEntity
from ote_sdk.entities.model_template import (
    TaskType,
    parse_model_template,
    task_type_to_label_domain,
)
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.entities.shapes.ellipse import Ellipse
from ote_sdk.entities.shapes.polygon import Point, Polygon
from ote_sdk.entities.shapes.rectangle import Rectangle
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.entities.train_parameters import TrainParameters
from ote_sdk.test_suite.e2e_test_system import e2e_pytest_api
from ote_sdk.tests.test_helpers import generate_random_annotated_image
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType
from ote_sdk.utils.shape_factory import ShapeFactory

DEFAULT_CLS_TEMPLATE_DIR = osp.join('configs', 'classification', 'efficientnet_b0_cls_incr')
DEFAULT_DET_TEMPLATE_DIR = osp.join('configs', 'detection', 'mobilenetv2_atss_cls_incr')
DEFAULT_SEG_TEMPLATE_DIR = osp.join('configs', 'segmentation', 'ocr-lite-hrnet-18-cls-incr')


def eval(task: BaseTask, model: ModelEntity, dataset: DatasetEntity) -> Performance:
    start_time = time.time()
    result_dataset = task.infer(dataset.with_empty_annotations())
    end_time = time.time()
    print(f'{len(dataset)} analysed in {end_time - start_time} seconds')
    result_set = ResultSetEntity(
        model=model,
        ground_truth_dataset=dataset,
        prediction_dataset=result_dataset
    )
    task.evaluate(result_set)
    assert result_set.performance is not None
    return result_set.performance


class MPAClsAPI(unittest.TestCase):
    @e2e_pytest_api
    def test_reading_classification_cls_incr_model_template(self):
        classification_template = ['efficientnet_b0_cls_incr', 'efficientnet_v2_s_cls_incr',
                                   'mobilenet_v3_large_1_cls_incr', 'mobilenet_v3_large_075_cls_incr',
                                   'mobilenet_v3_small_cls_incr']
        for model_template in classification_template:
            parse_model_template(osp.join('configs', 'classification', model_template, 'template_experimental.yaml'))

    @staticmethod
    def generate_label_schema(not_empty_labels, multilabel=False):
        assert len(not_empty_labels) > 1

        label_schema = LabelSchemaEntity()
        if multilabel:
            emptylabel = LabelEntity(name="Empty label", is_empty=True, domain=Domain.CLASSIFICATION)
            empty_group = LabelGroup(name="empty", labels=[emptylabel], group_type=LabelGroupType.EMPTY_LABEL)
            for label in not_empty_labels:
                label_schema.add_group(LabelGroup(name=label.name, labels=[label], group_type=LabelGroupType.EXCLUSIVE))
            label_schema.add_group(empty_group)
        else:
            main_group = LabelGroup(name="labels", labels=not_empty_labels, group_type=LabelGroupType.EXCLUSIVE)
            label_schema.add_group(main_group)
        return label_schema

    @staticmethod
    def setup_configurable_parameters(template_dir, num_iters=10):
        model_template = parse_model_template(osp.join(template_dir, 'template_experimental.yaml'))
        hyper_parameters = create(model_template.hyper_parameters.data)
        hyper_parameters.learning_parameters.num_iters = num_iters
        return hyper_parameters, model_template

    def init_environment(self, params, model_template, number_of_images=10):
        resolution = (224, 224)
        colors = [(0,255,0), (0,0,255)]
        cls_names = ['b', 'g']
        texts = ['Blue', 'Green']
        env_labels = [LabelEntity(name=name, domain=Domain.CLASSIFICATION, is_empty=False, id=ID(i)) for i, name in
                    enumerate(cls_names)]

        items = []

        for _ in range(0, number_of_images):
            for j, lbl in enumerate(env_labels):
                class_img = np.zeros((*resolution, 3), dtype=np.uint8)
                class_img[:] = colors[j]
                class_img = cv.putText(class_img, texts[j], (50, 50), cv.FONT_HERSHEY_SIMPLEX,
                                    .8 + j*.2, colors[j - 1], 2, cv.LINE_AA)

                image = Image(data=class_img)
                labels = [ScoredLabel(label=lbl, probability=1.0)]
                shapes = [Annotation(Rectangle.generate_full_box(), labels)]
                annotation_scene = AnnotationSceneEntity(kind=AnnotationSceneKind.ANNOTATION,
                                                        annotations=shapes)
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
        labels_schema = self.generate_label_schema(dataset.get_labels(), multilabel=False)
        environment = TaskEnvironment(model=None, hyper_parameters=params, label_schema=labels_schema,
                                    model_template=model_template)
        return environment, dataset

    @e2e_pytest_api
    def test_training_progress_tracking(self):
        print('Task initialized, model training starts.')
        training_progress_curve = []
        hyper_parameters, model_template = self.setup_configurable_parameters(DEFAULT_CLS_TEMPLATE_DIR, num_iters=5)
        task_environment, dataset = self.init_environment(hyper_parameters, model_template, 20)
        task = ClassificationTrainTask(task_environment=task_environment)
        task._delete_scratch_space()

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
        training_progress_curve = np.asarray(training_progress_curve)
        print(training_progress_curve)
        assert np.all(training_progress_curve[1:] >= training_progress_curve[:-1])

    @e2e_pytest_api
    def test_inference_progress_tracking(self):
        hyper_parameters, model_template = self.setup_configurable_parameters(DEFAULT_CLS_TEMPLATE_DIR, num_iters=5)
        task_environment, dataset = self.init_environment(hyper_parameters, model_template, 20)
        task = ClassificationInferenceTask(task_environment=task_environment)
        task._delete_scratch_space()

        print('Task initialized, model inference starts.')
        inference_progress_curve = []

        def progress_callback(progress: int):
            inference_progress_curve.append(progress)

        inference_parameters = InferenceParameters
        inference_parameters.update_progress = progress_callback

        task.infer(dataset.with_empty_annotations(), inference_parameters)

        assert len(inference_progress_curve) > 0
        inference_progress_curve = np.asarray(inference_progress_curve)
        assert np.all(inference_progress_curve[1:] >= inference_progress_curve[:-1])

    @e2e_pytest_api
    def test_inference_task(self):
        # Prepare pretrained weights
        hyper_parameters, model_template = self.setup_configurable_parameters(DEFAULT_CLS_TEMPLATE_DIR, num_iters=2)
        classification_environment, dataset = self.init_environment(hyper_parameters, model_template, 50)
        val_dataset = dataset.get_subset(Subset.VALIDATION)

        train_task = ClassificationTrainTask(task_environment=classification_environment)
        self.addCleanup(train_task._delete_scratch_space)

        trained_model = ModelEntity(
            dataset,
            classification_environment.get_model_configuration(),
        )
        train_task.train(dataset, trained_model, TrainParameters)
        performance_after_train = eval(train_task, trained_model, val_dataset)

        # Create InferenceTask
        classification_environment.model = trained_model
        inference_task = ClassificationInferenceTask(task_environment=classification_environment)
        self.addCleanup(inference_task._delete_scratch_space)

        performance_after_load = eval(inference_task, trained_model, val_dataset)

        assert performance_after_train == performance_after_load

        # Export
        exported_model = ModelEntity(
            dataset,
            classification_environment.get_model_configuration(),
            _id=ObjectId())
        inference_task.export(ExportType.OPENVINO, exported_model)


class MPADetAPI(unittest.TestCase):
    """
    Collection of tests for OTE API and OTE Model Templates
    """
    @e2e_pytest_api
    def test_reading_detection_cls_incr_model_template(self):
        detection_template = ['mobilenetv2_atss_cls_incr', 'resnet50_vfnet_cls_incr']
        for model_template in detection_template:
            parse_model_template(osp.join('configs', 'detection', model_template, 'template_experimental.yaml'))

    def init_environment(
            self,
            params,
            model_template,
            number_of_images=500,
            task_type=TaskType.DETECTION):

        labels_names = ('rectangle', 'ellipse', 'triangle')
        labels_schema = generate_label_schema(labels_names, task_type_to_label_domain(task_type))
        labels_list = labels_schema.get_labels(False)
        environment = TaskEnvironment(model=None, hyper_parameters=params, label_schema=labels_schema,
                                      model_template=model_template)

        warnings.filterwarnings('ignore', message='.* coordinates .* are out of bounds.*')
        items = []
        for i in range(0, number_of_images):
            image_numpy, annos = generate_random_annotated_image(
                image_width=640,
                image_height=480,
                labels=labels_list,
                max_shapes=20,
                min_size=50,
                max_size=100,
                random_seed=None)
            # Convert shapes according to task
            for anno in annos:
                if task_type == TaskType.INSTANCE_SEGMENTATION:
                    anno.shape = ShapeFactory.shape_as_polygon(anno.shape)
                else:
                    anno.shape = ShapeFactory.shape_as_rectangle(anno.shape)

            image = Image(data=image_numpy)
            annotation_scene = AnnotationSceneEntity(
                kind=AnnotationSceneKind.ANNOTATION,
                annotations=annos)
            items.append(DatasetItemEntity(media=image, annotation_scene=annotation_scene))
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
        glb = glob.glob(f'{template_dir}/template*.yaml')
        template_path = glb[0] if glb else None
        if not template_path:
          raise RuntimeError(f"Template YAML not found: {template_dir}")

        model_template = parse_model_template(template_path)
        hyper_parameters = create(model_template.hyper_parameters.data)
        hyper_parameters.learning_parameters.num_iters = num_iters
        hyper_parameters.postprocessing.result_based_confidence_threshold = False
        hyper_parameters.postprocessing.confidence_threshold = 0.1
        return hyper_parameters, model_template

    @e2e_pytest_api
    def test_cancel_training_detection(self):
        """
        Tests starting and cancelling training.

        Flow of the test:
        - Creates a randomly annotated project with a small dataset containing 3 classes:
            ['rectangle', 'triangle', 'circle'].
        - Start training and give cancel training signal after 10 seconds. Assert that training
            stops within 35 seconds after that
        - Start training and give cancel signal immediately. Assert that training stops within 25 seconds.

        This test should be finished in under one minute on a workstation.
        """
        hyper_parameters, model_template = self.setup_configurable_parameters(DEFAULT_DET_TEMPLATE_DIR, num_iters=500)
        detection_environment, dataset = self.init_environment(hyper_parameters, model_template, 64)

        detection_task = DetectionTrainTask(task_environment=detection_environment)

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='train_thread')

        output_model = ModelEntity(
            dataset,
            detection_environment.get_model_configuration(),
        )

        training_progress_curve = []
        def progress_callback(progress: float, score: Optional[float] = None):
            training_progress_curve.append(progress)

        train_parameters = TrainParameters
        train_parameters.update_progress = progress_callback

        # Test stopping after some time
        start_time = time.time()
        train_future = executor.submit(detection_task.train, dataset, output_model, train_parameters)
        # give train_thread some time to initialize the model
        while not detection_task._is_training:
            time.sleep(10)
        detection_task.cancel_training()

        # stopping process has to happen in less than 35 seconds
        train_future.result()
        self.assertEqual(training_progress_curve[-1], 100)
        self.assertLess(time.time() - start_time, 100, 'Expected to stop within 100 seconds.')

        # Test stopping immediately
        start_time = time.time()
        train_future = executor.submit(detection_task.train, dataset, output_model)
        detection_task.cancel_training()

        train_future.result()
        self.assertLess(time.time() - start_time, 25)  # stopping process has to happen in less than 25 seconds

    @e2e_pytest_api
    def test_training_progress_tracking(self):
        hyper_parameters, model_template = self.setup_configurable_parameters(DEFAULT_DET_TEMPLATE_DIR, num_iters=5)
        detection_environment, dataset = self.init_environment(hyper_parameters, model_template, 50)

        task = DetectionTrainTask(task_environment=detection_environment)
        self.addCleanup(task._delete_scratch_space)

        print('Task initialized, model training starts.')
        training_progress_curve = []

        def progress_callback(progress: float, score: Optional[float] = None):
            training_progress_curve.append(progress)

        train_parameters = TrainParameters
        train_parameters.update_progress = progress_callback
        output_model = ModelEntity(
            dataset,
            detection_environment.get_model_configuration(),
        )
        task.train(dataset, output_model, train_parameters)

        self.assertGreater(len(training_progress_curve), 0)
        training_progress_curve = np.asarray(training_progress_curve)
        self.assertTrue(np.all(training_progress_curve[1:] >= training_progress_curve[:-1]))

    @e2e_pytest_api
    def test_inference_progress_tracking(self):
        hyper_parameters, model_template = self.setup_configurable_parameters(DEFAULT_DET_TEMPLATE_DIR, num_iters=10)
        detection_environment, dataset = self.init_environment(hyper_parameters, model_template, 50)

        task = DetectionInferenceTask(task_environment=detection_environment)
        self.addCleanup(task._delete_scratch_space)

        print('Task initialized, model inference starts.')
        inference_progress_curve = []

        def progress_callback(progress: int):
            assert isinstance(progress, int)
            inference_progress_curve.append(progress)

        inference_parameters = InferenceParameters
        inference_parameters.update_progress = progress_callback

        task.infer(dataset.with_empty_annotations(), inference_parameters)

        self.assertGreater(len(inference_progress_curve), 0)
        inference_progress_curve = np.asarray(inference_progress_curve)
        self.assertTrue(np.all(inference_progress_curve[1:] >= inference_progress_curve[:-1]))

    @e2e_pytest_api
    def test_inference_task(self):
        # Prepare pretrained weights
        hyper_parameters, model_template = self.setup_configurable_parameters(DEFAULT_DET_TEMPLATE_DIR, num_iters=2)
        detection_environment, dataset = self.init_environment(hyper_parameters, model_template, 50)
        val_dataset = dataset.get_subset(Subset.VALIDATION)

        train_task = DetectionTrainTask(task_environment=detection_environment)
        self.addCleanup(train_task._delete_scratch_space)

        trained_model = ModelEntity(
            dataset,
            detection_environment.get_model_configuration(),
        )
        train_task.train(dataset, trained_model, TrainParameters)
        performance_after_train = eval(train_task, trained_model, val_dataset)

        # Create InferenceTask
        detection_environment.model = trained_model
        inference_task = DetectionInferenceTask(task_environment=detection_environment)
        self.addCleanup(inference_task._delete_scratch_space)

        performance_after_load = eval(inference_task, trained_model, val_dataset)

        assert performance_after_train == performance_after_load

        # Export
        exported_model = ModelEntity(
            dataset,
            detection_environment.get_model_configuration(),
            _id=ObjectId())
        inference_task.export(ExportType.OPENVINO, exported_model)


class MPASegAPI(unittest.TestCase):
    """
    Collection of tests for OTE API and OTE Model Templates
    """
    @e2e_pytest_api
    def test_reading_segmentation_cls_incr_model_template(self):
        segmentation_template = ['ocr-lite-hrnet-18-cls-incr']
        for model_template in segmentation_template:
            parse_model_template(osp.join('configs', 'segmentation', model_template, 'template_experimental.yaml'))

    @staticmethod
    def generate_label_schema(label_names):
        label_domain = Domain.SEGMENTATION
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
        model_template = parse_model_template(osp.join(template_dir, 'template_experimental.yaml'))

        hyper_parameters = create(model_template.hyper_parameters.data)
        hyper_parameters.learning_parameters.learning_rate_fixed_iters = 0
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

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='train_thread')

        output_model = ModelEntity(
            dataset,
            segmentation_environment.get_model_configuration(),
        )

        training_progress_curve = []
        def progress_callback(progress: float, score: Optional[float] = None):
            training_progress_curve.append(progress)

        train_parameters = TrainParameters
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
        self.assertEqual(training_progress_curve[-1], 100)
        self.assertLess(time.time() - start_time, 100, 'Expected to stop within 100 seconds.')

        # Test stopping immediately
        start_time = time.time()
        train_future = executor.submit(segmentation_task.train, dataset, output_model)
        segmentation_task.cancel_training()

        train_future.result()
        self.assertLess(time.time() - start_time, 25)  # stopping process has to happen in less than 25 seconds

    @e2e_pytest_api
    def test_training_progress_tracking(self):
        hyper_parameters, model_template = self.setup_configurable_parameters(DEFAULT_SEG_TEMPLATE_DIR, num_iters=5)
        segmentation_environment, dataset = self.init_environment(hyper_parameters, model_template, 12)

        task = SegmentationTrainTask(task_environment=segmentation_environment)
        #self.addCleanup(task._delete_scratch_space)

        print('Task initialized, model training starts.')
        training_progress_curve = []

        def progress_callback(progress: float, score: Optional[float] = None):
            training_progress_curve.append(progress)

        train_parameters = TrainParameters
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
        hyper_parameters, model_template = self.setup_configurable_parameters(DEFAULT_SEG_TEMPLATE_DIR, num_iters=10)
        segmentation_environment, dataset = self.init_environment(hyper_parameters, model_template, 12)

        task = SegmentationInferenceTask(task_environment=segmentation_environment)
        self.addCleanup(task._delete_scratch_space)

        print('Task initialized, model inference starts.')
        inference_progress_curve = []

        def progress_callback(progress: int):
            assert isinstance(progress, int)
            inference_progress_curve.append(progress)

        inference_parameters = InferenceParameters
        inference_parameters.update_progress = progress_callback

        task.infer(dataset.with_empty_annotations(), inference_parameters)

        self.assertGreater(len(inference_progress_curve), 0)
        inference_progress_curve = np.asarray(inference_progress_curve)
        self.assertTrue(np.all(inference_progress_curve[1:] >= inference_progress_curve[:-1]))

    @e2e_pytest_api
    def test_inference_task(self):
        # Prepare pretrained weights
        hyper_parameters, model_template = self.setup_configurable_parameters(DEFAULT_SEG_TEMPLATE_DIR, num_iters=2)
        segmentation_environment, dataset = self.init_environment(hyper_parameters, model_template, 30)
        val_dataset = dataset.get_subset(Subset.VALIDATION)

        train_task = SegmentationTrainTask(task_environment=segmentation_environment)
        self.addCleanup(train_task._delete_scratch_space)

        trained_model = ModelEntity(
            dataset,
            segmentation_environment.get_model_configuration(),
        )
        train_task.train(dataset, trained_model, TrainParameters)
        performance_after_train = eval(train_task, trained_model, val_dataset)

        # Create InferenceTask
        segmentation_environment.model = trained_model
        inference_task = SegmentationInferenceTask(task_environment=segmentation_environment)
        self.addCleanup(inference_task._delete_scratch_space)

        performance_after_load = eval(inference_task, trained_model, val_dataset)

        assert performance_after_train == performance_after_load

        # Export
        exported_model = ModelEntity(
            dataset,
            segmentation_environment.get_model_configuration(),
            _id=ObjectId())
        inference_task.export(ExportType.OPENVINO, exported_model)
