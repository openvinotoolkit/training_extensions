import numpy as np
import pytest

from otx.algorithms.classification.adapters.mmcls.data.datasets import SelfSLDataset
from otx.api.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.image import Image
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.rectangle import Rectangle
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.api.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)


def create_cls_dataset():
    image = Image(data=np.random.randint(low=0, high=255, size=(8, 8, 3)))
    annotation = Annotation(
        shape=Rectangle.generate_full_box(),
        labels=[ScoredLabel(LabelEntity(name="test_selfsl_dataset", domain=Domain.CLASSIFICATION))],
    )
    annotation_scene = AnnotationSceneEntity(annotations=[annotation], kind=AnnotationSceneKind.ANNOTATION)
    dataset_item = DatasetItemEntity(media=image, annotation_scene=annotation_scene)

    dataset = DatasetEntity(items=[dataset_item])
    return dataset, dataset.get_labels()


class TestSelfSLDataset:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.otx_dataset, _ = create_cls_dataset()
        self.pipeline = {
            "view0": [dict(type="ImageToTensor", keys=["img"])],
            "view1": [dict(type="ImageToTensor", keys=["img"])],
        }

    @e2e_pytest_unit
    def test_self_sl_dataset_init_params_validation(self):
        """Test SelfSLDataset initialization parameters validation."""
        correct_values_dict = {
            "otx_dataset": self.otx_dataset,
            "pipeline": self.pipeline,
        }
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "otx_dataset" parameter
            ("otx_dataset", unexpected_str),
            # Unexpected string is specified as "pipeline" parameter
            ("pipeline", unexpected_str),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=SelfSLDataset,
        )

    @e2e_pytest_unit
    def test_getitem(self):
        """Test __getitem__ method."""
        dataset = SelfSLDataset(otx_dataset=self.otx_dataset, pipeline=self.pipeline)

        data_item = dataset[0]
        for i in range(1, 3):
            assert f"dataset_item{i}" in data_item
            assert f"width{i}" in data_item
            assert f"height{i}" in data_item
            assert f"index{i}" in data_item
            assert f"filename{i}" in data_item
            assert f"ori_filename{i}" in data_item
            assert f"img{i}" in data_item
            assert f"img_shape{i}" in data_item
            assert f"ori_shape{i}" in data_item
            assert f"pad_shape{i}" in data_item
            assert f"img_norm_cfg{i}" in data_item
            assert f"img_fields{i}" in data_item
