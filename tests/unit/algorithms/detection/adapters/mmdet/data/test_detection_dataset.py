# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import random
import warnings

import pytest

from otx.algorithms.detection.adapters.mmdet.data import MPADetDataset
from otx.algorithms.detection.utils import generate_label_schema
from otx.api.entities.annotation import AnnotationSceneEntity, AnnotationSceneKind
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.image import Image
from otx.api.entities.label import Domain
from otx.api.entities.model_template import TaskType, task_type_to_label_domain
from otx.api.entities.subset import Subset
from otx.api.utils.shape_factory import ShapeFactory
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.api.test_helpers import generate_random_annotated_image


def generate_fake_det_dataset(number_of_images=1, task_type=TaskType.DETECTION):

    labels_names = ("rectangle", "ellipse", "triangle")
    labels_schema = generate_label_schema(labels_names, task_type_to_label_domain(task_type))
    labels_list = labels_schema.get_labels(False)

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
            random_seed=None,
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
    return dataset, dataset.get_labels()


class TestOTXDetDataset:
    """Check Builder's function is working well.

    1. Check "Builder.build_task_config" function that create otx-workspace is working well.
    <Steps>
        1. Create Classification custom workspace
        2. Raising Error of building workspace with already created path
        3. Update hparam.yaml with train_type="selfsl"
        4. Raising ValueError with wrong train_type
        5. Build workspace with model_type argments
        6. Raise ValueError when build workspace with wrong model_type argments

    2. Check "Builder.build_backbone_config" function that generate backbone configuration file is working well
    <Steps>
        1. Generate backbone config file (mmcls.MMOVBackbone)
        2. Raise ValueError with wrong output_path

    3. Check "Builder.merge_backbone" function that update model config with new backbone is working well
    <Steps>
        1. Update model config with mmcls.ResNet backbone (default model.backbone: otx.OTXEfficientNet)
        2. Raise ValueError with wrong model_config_path
        3. Raise ValueError with wrong backbone_config_path
        4. Update model config without backbone's out_indices
        5. Update model config with backbone's pretrained path
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.otx_dataset, self.labels = generate_fake_det_dataset()
        self.pipeline = []
        self.domain = Domain.DETECTION

    # def test_init_dataset(self) -> None:
    # 이상한거 들어오면 오류나게

    @e2e_pytest_unit
    def test_prepare_train_img(self) -> None:
        """Create Classification custom workspace."""
        dataset = MPADetDataset(self.otx_dataset, self.labels, self.pipeline, self.domain)
        breakpoint()
        img = dataset.prepare_train_img(0)
        assert type(img) == dict

    @e2e_pytest_unit
    def test_pre_pipeline(self) -> None:
        """Create Classification custom workspace."""
        results = dict()
        MPADetDataset.pre_pipeline(results)
        assert "bbox_fields" in results
        assert "mask_fields" in results
        assert "seg_fields" in results

    @e2e_pytest_unit
    def test_prepare_train_img_out_of_index(self) -> None:
        """Create Classification custom workspace."""
        dataset = MPADetDataset(self.otx_dataset, self.labels, self.pipeline, self.domain)
        with pytest.raises(IndexError):
            dataset.prepare_train_img(5000)

    @e2e_pytest_unit
    def test_prepare_test_img(self) -> None:
        """Create Classification custom workspace."""
        dataset = MPADetDataset(self.otx_dataset, self.labels, self.pipeline, self.domain)
        img = dataset.prepare_test_img(0)
        assert type(img) == dict

    @e2e_pytest_unit
    def test_get_ann_info(self) -> None:
        """Create Classification custom workspace."""
        dataset = MPADetDataset(self.otx_dataset, self.labels, self.pipeline, self.domain)
        ann_info = dataset.get_ann_info(0)
        assert type(ann_info) == dict
        assert "bboxes" in ann_info
        assert "labels" in ann_info

    # @e2e_pytest_unit
    # def test_evaluate(self) -> None:
    #     """Create Classification custom workspace."""
    #     ann_info = self.dataset.evaluate(results)
    #     assert type(ann_info) == dict
    #     assert "bboxes" in ann_info
    #     assert "labels" in ann_info
