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

# pylint:disable=redefined-outer-name, protected-access

import os.path as osp
import random
from typing import Optional

import cv2 as cv
import numpy as np
import pytest
from e2e_test_system import e2e_pytest_api

from ote_sdk.entities.annotation import Annotation, AnnotationSceneEntity, AnnotationSceneKind
from ote_sdk.entities.datasets import Subset, DatasetEntity
from ote_sdk.entities.dataset_item import DatasetItemEntity
from ote_sdk.entities.id import ID
from ote_sdk.entities.image import Image
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.label import LabelEntity, Domain
from ote_sdk.entities.model import ModelEntity
from ote_sdk.entities.model_template import parse_model_template
from ote_sdk.entities.shapes.rectangle import Rectangle
from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.entities.train_parameters import TrainParameters
from ote_sdk.configuration.helper import convert, create

from torchreid_tasks.parameters import OTEClassificationParameters
from torchreid_tasks.train_task import OTEClassificationTrainingTask
from torchreid_tasks.utils import generate_label_schema


DEFAULT_TEMPLATE_DIR = osp.join('configs', 'ote_custom_classification', 'efficientnet_b0')


@e2e_pytest_api
def test_reading_efficientnet_b0():
    parse_model_template(osp.join('configs', 'ote_custom_classification', 'efficientnet_b0', 'template.yaml'))


@e2e_pytest_api
def test_reading_mobilenet_v3_large_075():
    parse_model_template(osp.join('configs', 'ote_custom_classification', 'mobilenet_v3_large_075', 'template_experimental.yaml'))


@e2e_pytest_api
def test_configuration_yaml():
    configuration = OTEClassificationParameters()
    configuration_yaml_str = convert(configuration, str)
    configuration_yaml_converted = create(configuration_yaml_str)
    configuration_yaml_loaded = create(osp.join('torchreid_tasks', 'configuration.yaml'))
    assert configuration_yaml_converted == configuration_yaml_loaded


def setup_configurable_parameters(template_dir, max_num_epochs=10):
    model_template = parse_model_template(osp.join(template_dir, 'template.yaml'))
    hyper_parameters = create(model_template.hyper_parameters.data)
    hyper_parameters.learning_parameters.max_num_epochs = max_num_epochs
    return hyper_parameters, model_template


def init_environment(params, model_template, number_of_images=10):
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
    labels_schema = generate_label_schema(dataset.get_labels(), multilabel=False)
    environment = TaskEnvironment(model=None, hyper_parameters=params, label_schema=labels_schema,
                                  model_template=model_template)
    return environment, dataset


@pytest.fixture()
def default_task_setup():
    hyper_parameters, model_template = setup_configurable_parameters(DEFAULT_TEMPLATE_DIR, max_num_epochs=5)
    task_environment, dataset = init_environment(hyper_parameters, model_template, 20)
    task = OTEClassificationTrainingTask(task_environment=task_environment)

    yield (task, task_environment, dataset)

    task._delete_scratch_space()


@e2e_pytest_api
def test_training_progress_tracking(default_task_setup):
    print('Task initialized, model training starts.')
    training_progress_curve = []
    task, task_environment, dataset = default_task_setup

    def progress_callback(progress: float, score: Optional[float] = None):
        training_progress_curve.append(progress)

    train_parameters = TrainParameters()
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
def test_inference_progress_tracking(default_task_setup):
    task, _, dataset = default_task_setup

    print('Task initialized, model inference starts.')
    inference_progress_curve = []

    def progress_callback(progress: int):
        inference_progress_curve.append(progress)

    inference_parameters = InferenceParameters()
    inference_parameters.update_progress = progress_callback

    task.infer(dataset.with_empty_annotations(), inference_parameters)

    assert len(inference_progress_curve) > 0
    inference_progress_curve = np.asarray(inference_progress_curve)
    assert np.all(inference_progress_curve[1:] >= inference_progress_curve[:-1])

@e2e_pytest_api
def test_nncf_optimize_progress_tracking():
    pytest.xfail('NNCF is not supported yet')
