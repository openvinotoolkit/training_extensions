"""API Tests for detection training"""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import glob
import warnings
import random
import os.path as osp

from otx.algorithms.detection.utils import generate_label_schema
from otx.api.configuration.helper import create
from otx.api.entities.annotation import AnnotationSceneEntity, AnnotationSceneKind
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.image import Image
from otx.api.entities.model_template import (
    TaskType,
    parse_model_template,
    task_type_to_label_domain,
)
from otx.api.entities.subset import Subset
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.utils.shape_factory import ShapeFactory
from tests.test_helpers import generate_random_annotated_image

DEFAULT_DET_TEMPLATE_DIR = osp.join("src/otx/algorithms/detection/configs", "detection", "mobilenetv2_atss")


class DetectionTaskAPIBase:
    """
    Collection of tests for OTX API and OTX Model Templates
    """

    def init_environment(self, params, model_template, number_of_images=500, task_type=TaskType.DETECTION):

        labels_names = ("rectangle", "ellipse", "triangle")
        labels_schema = generate_label_schema(labels_names, task_type_to_label_domain(task_type))
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
            image_numpy, annos = generate_random_annotated_image(
                image_width=640,
                image_height=480,
                labels=labels_list,
                max_shapes=20,
                min_size=50,
                max_size=100,
            )
            # Convert shapes according to task
            for anno in annos:
                if task_type == TaskType.INSTANCE_SEGMENTATION:
                    anno.shape = ShapeFactory.shape_as_polygon(anno.shape)
                else:
                    anno.shape = ShapeFactory.shape_as_rectangle(anno.shape)

            image = Image(data=image_numpy)
            annotation_scene = AnnotationSceneEntity(kind=AnnotationSceneKind.ANNOTATION, annotations=annos)
            items.append(DatasetItemEntity(media=image, annotation_scene=annotation_scene))
        warnings.resetwarnings()

        rng = random.Random()  # nosec B311 used random for testing only
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
    def setup_configurable_parameters(template_dir, num_iters=10, tiling=False):
        glb = glob.glob(f"{template_dir}/template*.yaml")
        template_path = glb[0] if glb else None
        if not template_path:
            raise RuntimeError(f"Template YAML not found: {template_dir}")

        model_template = parse_model_template(template_path)
        hyper_parameters = create(model_template.hyper_parameters.data)
        hyper_parameters.learning_parameters.num_iters = num_iters
        hyper_parameters.postprocessing.result_based_confidence_threshold = False
        hyper_parameters.postprocessing.confidence_threshold = 0.1
        if tiling:
            hyper_parameters.tiling_parameters.enable_tiling = True
        return hyper_parameters, model_template
