# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import glob
import io
import os.path as osp
import random
import time
import unittest
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import torch
from bson import ObjectId
from ote_sdk.test_suite.e2e_test_system import e2e_pytest_api
from ote_sdk.configuration.helper import create
from ote_sdk.entities.annotation import Annotation, AnnotationSceneEntity, AnnotationSceneKind
from ote_sdk.entities.dataset_item import DatasetItemEntity
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.image import Image
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.color import Color
from ote_sdk.entities.label import Domain, LabelEntity
from ote_sdk.entities.label_schema import LabelGroup, LabelGroupType, LabelSchemaEntity
from ote_sdk.entities.model import ModelEntity, ModelFormat, ModelOptimizationType
from ote_sdk.entities.model_template import TaskType, parse_model_template, task_type_to_label_domain
from ote_sdk.entities.metrics import Performance
from ote_sdk.entities.optimization_parameters import OptimizationParameters
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.shapes.ellipse import Ellipse
from ote_sdk.entities.shapes.polygon import Polygon, Point
from ote_sdk.entities.shapes.rectangle import Rectangle
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.entities.train_parameters import TrainParameters
from ote_sdk.tests.test_helpers import generate_random_annotated_image
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType, IExportTask
from ote_sdk.usecases.tasks.interfaces.optimization_interface import OptimizationType
from ote_sdk.utils.shape_factory import ShapeFactory

from detection_tasks.apis.detection import OpenVINODetectionTask
from detection_tasks.apis.detection.ote_utils import generate_label_schema
from mpa_tasks.apis.detection import DetectionTrainTask, DetectionInferenceTask
from mpa_tasks.apis.segmentation import SegmentationTrainTask

DEFAULT_DET_TEMPLATE_DIR = osp.join('configs', 'detection', 'mobilenetv2_atss_cls_incr')
DEFAULT_SEG_TEMPLATE_DIR = osp.join('configs', 'segmentation', 'ocr-lite-hrnet-18-cls-incr')

class MPADetAPI(unittest.TestCase):
    """
    Collection of tests for OTE API and OTE Model Templates
    """

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

    def setup_configurable_parameters(self, template_dir, num_iters=10):
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

    # @e2e_pytest_api
    # def test_cancel_training_detection(self):
    #     """
    #     Tests starting and cancelling training.

    #     Flow of the test:
    #     - Creates a randomly annotated project with a small dataset containing 3 classes:
    #         ['rectangle', 'triangle', 'circle'].
    #     - Start training and give cancel training signal after 10 seconds. Assert that training
    #         stops within 35 seconds after that
    #     - Start training and give cancel signal immediately. Assert that training stops within 25 seconds.

    #     This test should be finished in under one minute on a workstation.
    #     """
    #     hyper_parameters, model_template = self.setup_configurable_parameters(DEFAULT_DET_TEMPLATE_DIR, num_iters=500)
    #     detection_environment, dataset = self.init_environment(hyper_parameters, model_template, 64)

    #     detection_task = DetectionTrainTask(task_environment=detection_environment)

    #     executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='train_thread')

    #     output_model = ModelEntity(
    #         dataset,
    #         detection_environment.get_model_configuration(),
    #     )

    #     training_progress_curve = []
    #     def progress_callback(progress: float, score: Optional[float] = None):
    #         training_progress_curve.append(progress)

    #     train_parameters = TrainParameters
    #     train_parameters.update_progress = progress_callback

    #     # Test stopping after some time
    #     start_time = time.time()
    #     train_future = executor.submit(detection_task.train, dataset, output_model, train_parameters)
    #     # give train_thread some time to initialize the model
    #     while not detection_task._is_training:
    #         time.sleep(10)
    #     detection_task.cancel_training()

    #     # stopping process has to happen in less than 35 seconds
    #     train_future.result()
    #     self.assertEqual(training_progress_curve[-1], 100)
    #     self.assertLess(time.time() - start_time, 100, 'Expected to stop within 100 seconds.')

    #     # Test stopping immediately (as soon as training is started).
    #     start_time = time.time()
    #     train_future = executor.submit(detection_task.train, dataset, output_model)
    #     while not detection_task._is_training:
    #         time.sleep(0.1)
    #     detection_task.cancel_training()

    #     train_future.result()
    #     self.assertLess(time.time() - start_time, 25)  # stopping process has to happen in less than 25 seconds

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

        task = DetectionTrainTask(task_environment=detection_environment)
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
        performance_after_train = self.eval(train_task, trained_model, val_dataset)

        # Create InferenceTask
        detection_environment.model = trained_model
        inference_task = DetectionInferenceTask(task_environment=detection_environment)
        self.addCleanup(inference_task._delete_scratch_space)

        performance_after_load = self.eval(inference_task, trained_model, val_dataset)

        assert performance_after_train == performance_after_load

        # Export
        exported_model = ModelEntity(
            dataset,
            detection_environment.get_model_configuration(),
            _id=ObjectId())
        inference_task.export(ExportType.OPENVINO, exported_model)

    @staticmethod
    def eval(task: DetectionTrainTask, model: ModelEntity, dataset: DatasetEntity) -> Performance:
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

    def check_threshold(self, reference, value, delta_tolerance, message=''):
        delta = value.score.value - reference.score.value
        self.assertLessEqual(
            np.abs(delta),
            delta_tolerance,
            msg=message +
                f' (reference metric: {reference.score.value}, '
                f'actual value: {value.score.value}, '
                f'delta tolerance threshold: {delta_tolerance})'
            )

    def end_to_end(
            self,
            template_dir,
            num_iters=5,
            quality_score_threshold=0.5,
            reload_perf_delta_tolerance=0.0,
            export_perf_delta_tolerance=0.01,
            pot_perf_delta_tolerance=0.1,
            nncf_perf_delta_tolerance=0.1,
            task_type=TaskType.DETECTION):

        hyper_parameters, model_template = self.setup_configurable_parameters(
            template_dir, num_iters=num_iters)
        detection_environment, dataset = self.init_environment(
            hyper_parameters, model_template, 250, task_type=task_type)

        val_dataset = dataset.get_subset(Subset.VALIDATION)
        task = DetectionTrainTask(task_environment=detection_environment)
        self.addCleanup(task._delete_scratch_space)

        print('Task initialized, model training starts.')
        # Train the task.
        # train_task checks that the task returns an Model and that
        # validation f-measure is higher than the threshold, which is a pretty low bar
        # considering that the dataset is so easy
        output_model = ModelEntity(
            dataset,
            detection_environment.get_model_configuration(),
            _id=ObjectId())
        task.train(dataset, output_model)

        # Test that output model is valid.
        modelinfo = torch.load(io.BytesIO(output_model.get_data("weights.pth")))
        modelinfo.pop('anchors', None)
        self.assertEqual(list(modelinfo.keys()), ['model', 'config', 'confidence_threshold', 'VERSION'])

        # Run inference.
        validation_performance = self.eval(task, output_model, val_dataset)
        print(f'Performance: {validation_performance.score.value:.4f}')
        self.assertGreater(validation_performance.score.value, quality_score_threshold,
            f'Expected F-measure to be higher than {quality_score_threshold}')

        # Run another training round.
        first_model = output_model
        new_model = ModelEntity(
            dataset,
            detection_environment.get_model_configuration(),
            _id=ObjectId())
        task._hyperparams.learning_parameters.num_iters = 1
        task.train(dataset, new_model)
        self.assertNotEqual(first_model, new_model)
        self.assertNotEqual(first_model.get_data("weights.pth"), new_model.get_data("weights.pth"))

        # Reload task with the first model.
        detection_environment.model = first_model
        task = DetectionTrainTask(detection_environment)
        self.assertEqual(task._task_environment.model.id, first_model.id)

        print('Reevaluating model.')
        # Performance should be the same after reloading
        performance_after_reloading = self.eval(task, output_model, val_dataset)
        print(f'Performance after reloading: {performance_after_reloading.score.value:.4f}')
        self.check_threshold(validation_performance, performance_after_reloading, reload_perf_delta_tolerance,
            'Too big performance difference after model reload.')

        if isinstance(task, IExportTask):
            # Run export.
            exported_model = ModelEntity(
                dataset,
                detection_environment.get_model_configuration(),
                _id=ObjectId())
            task.export(ExportType.OPENVINO, exported_model)
            self.assertEqual(exported_model.model_format, ModelFormat.OPENVINO)
            self.assertEqual(exported_model.optimization_type, ModelOptimizationType.MO)

            # Create OpenVINO Task and evaluate the model.
            detection_environment.model = exported_model
            ov_task = OpenVINODetectionTask(detection_environment)
            predicted_validation_dataset = ov_task.infer(val_dataset.with_empty_annotations())
            resultset = ResultSetEntity(
                model=output_model,
                ground_truth_dataset=val_dataset,
                prediction_dataset=predicted_validation_dataset,
            )
            ov_task.evaluate(resultset)
            export_performance = resultset.performance
            assert export_performance is not None
            print(f'Performance of exported model: {export_performance.score.value:.4f}')
            self.check_threshold(validation_performance, export_performance, export_perf_delta_tolerance,
                'Too big performance difference after OpenVINO export.')

            # Run POT optimization and evaluate the result.
            print('Run POT optimization.')
            optimized_model = ModelEntity(
                dataset,
                detection_environment.get_model_configuration(),
            )
            ov_task.optimize(OptimizationType.POT, dataset, optimized_model, OptimizationParameters())
            pot_performance = self.eval(ov_task, optimized_model, val_dataset)
            print(f'Performance of optimized model: {pot_performance.score.value:.4f}')
            self.check_threshold(validation_performance, pot_performance, pot_perf_delta_tolerance,
                'Too big performance difference after POT optimization.')


class MPASegAPI(unittest.TestCase):
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
        hyper_parameters.learning_parameters.learning_rate_warmup_iters = 1
        hyper_parameters.learning_parameters.num_iters = num_iters
        hyper_parameters.learning_parameters.num_checkpoints = 1

        return hyper_parameters, model_template

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

        task = SegmentationTrainTask(task_environment=segmentation_environment)
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
