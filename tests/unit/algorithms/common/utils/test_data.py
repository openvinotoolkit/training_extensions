"""Tests for data utils for common OTX algorithms."""
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from otx.algorithms.common.utils.data import (
    compute_robust_statistics,
    compute_robust_scale_statistics,
    compute_robust_dataset_statistics,
)
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.image import Image
from otx.api.entities.annotation import Annotation, AnnotationSceneEntity, AnnotationSceneKind
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.scored_label import ScoredLabel, LabelEntity, Domain

import numpy as np


@e2e_pytest_unit
def test_compute_robust_statistics():
    values = np.array([])
    stat = compute_robust_statistics(values)
    assert len(stat) == 0

    values = np.array([0.5, 1, 1.5])
    stat = compute_robust_statistics(values)
    assert np.isclose(stat["avg"], 1.0)
    assert np.isclose(stat["min"], 0.5)
    assert np.isclose(stat["max"], 1.5)

    values = np.random.rand(10)
    stat = compute_robust_statistics(values)
    assert np.isclose(stat["min"], np.min(values))
    assert np.isclose(stat["max"], np.max(values))
    assert stat["min"] <= stat["robust_min"]
    assert stat["max"] <= stat["robust_max"]


@e2e_pytest_unit
def test_compute_robust_scale_statistics():
    scales = np.array([])
    stat = compute_robust_scale_statistics(scales)
    assert len(stat) == 0

    scales = np.array([0.5, 1, 2])
    stat = compute_robust_scale_statistics(scales)
    assert np.isclose(stat["avg"], 1.0)
    assert np.isclose(stat["min"], 0.5)
    assert np.isclose(stat["max"], 2.0)

    scales = np.random.rand(10)
    stat = compute_robust_scale_statistics(scales)
    assert np.isclose(stat["min"], np.min(scales))
    assert np.isclose(stat["max"], np.max(scales))
    assert stat["min"] <= stat["robust_min"]
    assert stat["max"] <= stat["robust_max"]


@e2e_pytest_unit
def test_compute_robuste_dataset_statistics():
    dataset = DatasetEntity()
    stat = compute_robust_dataset_statistics(dataset)
    assert len(stat) == 0

    label = ScoredLabel(label=LabelEntity(name="test", domain=Domain.DETECTION))
    dataset = DatasetEntity(
        items=[
            DatasetItemEntity(
                Image(data=np.random.rand(50, 50)),
                AnnotationSceneEntity(
                    annotations=[
                        Annotation(shape=Rectangle(x1=0.0, y1=0.0, x2=0.1, y2=0.1), labels=[label]),
                    ],
                    kind=AnnotationSceneKind.ANNOTATION,
                ),
            ),
            DatasetItemEntity(
                Image(data=np.random.rand(100, 100)),
                AnnotationSceneEntity(
                    annotations=[
                        Annotation(shape=Rectangle(x1=0.0, y1=0.0, x2=0.1, y2=0.1), labels=[label]),
                        Annotation(shape=Rectangle(x1=0.1, y1=0.1, x2=0.3, y2=0.3), labels=[label]),
                    ],
                    kind=AnnotationSceneKind.ANNOTATION,
                ),
            ),
            DatasetItemEntity(
                Image(data=np.random.rand(200, 200)),
                AnnotationSceneEntity(
                    annotations=[],
                    kind=AnnotationSceneKind.ANNOTATION,
                ),
            ),
        ]
    )

    stat = compute_robust_dataset_statistics(dataset, max_samples=0)
    assert len(stat) == 0
    stat = compute_robust_dataset_statistics(dataset, max_samples=-1)
    assert len(stat) == 0

    stat = compute_robust_dataset_statistics(dataset, ann_stat=False)
    assert np.isclose(stat["image"]["avg"], 100)
    assert "annotation" not in stat

    stat = compute_robust_dataset_statistics(dataset, ann_stat=True)
    assert np.isclose(stat["annotation"]["num_per_image"]["avg"], 1.0)
    assert np.isclose(stat["annotation"]["size_of_shape"]["avg"], 10.0)
