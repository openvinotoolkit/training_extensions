# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


from datetime import datetime

import numpy as np
import pytest

from otx.api.configuration import ConfigurableParameters
from otx.api.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from otx.api.entities.color import Color
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.id import ID
from otx.api.entities.image import Image
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.label_schema import LabelGroup, LabelSchemaEntity
from otx.api.entities.metrics import (
    BarChartInfo,
    BarMetricsGroup,
    ColorPalette,
    MatrixMetric,
    Performance,
    ScoreMetric,
    VisualizationType,
)
from otx.api.entities.model import ModelConfiguration, ModelEntity
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.usecases.evaluation.accuracy import (
    Accuracy,
    compute_unnormalized_confusion_matrices_from_resultset,
    precision_metrics_group,
    recall_metrics_group,
)
from otx.api.usecases.evaluation.averaging import MetricAverageMethod
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements


class CommonActions:
    color = Color(0, 255, 0)
    creation_date = datetime(year=2021, month=12, day=27)
    image = Image(data=np.random.randint(low=0, high=255, size=(10, 16, 3)))

    def car(self) -> LabelEntity:
        return LabelEntity(
            name="car",
            domain=Domain.DETECTION,
            color=self.color,
            creation_date=self.creation_date,
            id=ID("car_label"),
        )

    def human(self) -> LabelEntity:
        return LabelEntity(
            name="human",
            domain=Domain.DETECTION,
            color=self.color,
            creation_date=self.creation_date,
            id=ID("human_label"),
        )

    def dog(self) -> LabelEntity:
        return LabelEntity(
            name="dog",
            domain=Domain.DETECTION,
            color=self.color,
            creation_date=self.creation_date,
            id=ID("dog_label"),
        )

    def human_1_dataset_item(self) -> DatasetItemEntity:
        human_1_roi = Annotation(shape=Rectangle.generate_full_box(), labels=[ScoredLabel(self.human())])
        human_1_annotation = Annotation(
            shape=Rectangle(x1=0.3, y1=0, x2=0.4, y2=0.2),
            labels=[ScoredLabel(self.human())],
        )
        human_1_annotation_scene = AnnotationSceneEntity(
            annotations=[human_1_annotation], kind=AnnotationSceneKind.ANNOTATION
        )
        human_1_dataset_item = DatasetItemEntity(
            media=self.image, annotation_scene=human_1_annotation_scene, roi=human_1_roi
        )
        return human_1_dataset_item

    def human_2_dataset_item(self) -> DatasetItemEntity:
        human_2_roi = Annotation(shape=Rectangle.generate_full_box(), labels=[ScoredLabel(self.human())])
        human_2_annotation = Annotation(
            shape=Rectangle(x1=0.6, y1=0, x2=0.7, y2=0.3),
            labels=[ScoredLabel(self.human())],
        )
        human_2_annotation_scene = AnnotationSceneEntity(
            annotations=[human_2_annotation], kind=AnnotationSceneKind.ANNOTATION
        )
        human_2_dataset_item = DatasetItemEntity(
            media=self.image, annotation_scene=human_2_annotation_scene, roi=human_2_roi
        )
        return human_2_dataset_item

    def human_3_dataset_item(self) -> DatasetItemEntity:
        human_3_roi = Annotation(shape=Rectangle.generate_full_box(), labels=[ScoredLabel(self.human())])
        human_3_annotation = Annotation(
            shape=Rectangle(x1=0.7, y1=0, x2=0.8, y2=0.3),
            labels=[ScoredLabel(self.human())],
        )
        human_3_annotation_scene = AnnotationSceneEntity(
            annotations=[human_3_annotation], kind=AnnotationSceneKind.ANNOTATION
        )
        human_3_dataset_item = DatasetItemEntity(
            media=self.image, annotation_scene=human_3_annotation_scene, roi=human_3_roi
        )
        return human_3_dataset_item

    def result_set(self) -> ResultSetEntity:
        configurable_params = ConfigurableParameters(header="Test model configurable params")
        labels_group = LabelGroup(
            name="model_labels_group",
            labels=[self.car(), self.human(), self.dog()],
        )
        other_label_group = LabelGroup(
            name="other_model_labels_group",
            labels=[self.human(), self.dog()],
        )
        model_configuration = ModelConfiguration(
            configurable_params,
            LabelSchemaEntity(label_groups=[labels_group, other_label_group]),
        )
        model = ModelEntity(train_dataset=DatasetEntity(), configuration=model_configuration)

        car_1_roi = Annotation(
            shape=Rectangle.generate_full_box(),
            labels=[ScoredLabel(self.car())],
        )
        car_1_annotation = Annotation(
            shape=Rectangle(x1=0, y1=0, x2=0.3, y2=0.2),
            labels=[ScoredLabel(self.car())],
        )
        car_1_annotation_scene = AnnotationSceneEntity(
            annotations=[car_1_annotation], kind=AnnotationSceneKind.ANNOTATION
        )
        car_1_dataset_item = DatasetItemEntity(media=self.image, annotation_scene=car_1_annotation_scene, roi=car_1_roi)

        car_2_roi = Annotation(shape=Rectangle.generate_full_box(), labels=[ScoredLabel(self.car())])
        car_2_annotation = Annotation(
            shape=Rectangle(x1=0.8, y1=0, x2=1.0, y2=0.2),
            labels=[ScoredLabel(self.car())],
        )
        car_2_annotation_scene = AnnotationSceneEntity(
            annotations=[car_2_annotation], kind=AnnotationSceneKind.ANNOTATION
        )
        car_2_dataset_item = DatasetItemEntity(media=self.image, annotation_scene=car_2_annotation_scene, roi=car_2_roi)

        dog_1_roi = Annotation(shape=Rectangle.generate_full_box(), labels=[ScoredLabel(self.dog())])
        dog_1_annotation = Annotation(
            shape=Rectangle(x1=0.5, y1=0, x2=0.6, y2=0.1),
            labels=[ScoredLabel(self.dog())],
        )
        dog_1_annotation_scene = AnnotationSceneEntity(
            annotations=[dog_1_annotation], kind=AnnotationSceneKind.ANNOTATION
        )
        dog_1_dataset_item = DatasetItemEntity(media=self.image, annotation_scene=dog_1_annotation_scene, roi=dog_1_roi)

        dog_2_roi = Annotation(shape=Rectangle.generate_full_box(), labels=[ScoredLabel(self.dog())])
        dog_2_annotation = Annotation(
            shape=Rectangle(x1=0.7, y1=0, x2=0.8, y2=0.3),
            labels=[ScoredLabel(self.dog())],
        )
        dog_2_annotation_scene = AnnotationSceneEntity(
            annotations=[dog_2_annotation], kind=AnnotationSceneKind.ANNOTATION
        )
        dog_2_dataset_item = DatasetItemEntity(media=self.image, annotation_scene=dog_2_annotation_scene, roi=dog_2_roi)

        ground_truth_dataset = DatasetEntity(
            [
                car_1_dataset_item,
                car_2_dataset_item,
                self.human_1_dataset_item(),
                self.human_2_dataset_item(),
                self.human_3_dataset_item(),
                dog_1_dataset_item,
            ]
        )
        prediction_dataset = DatasetEntity(
            [
                car_1_dataset_item,
                car_2_dataset_item,
                self.human_1_dataset_item(),
                self.human_2_dataset_item(),
                dog_1_dataset_item,
                dog_2_dataset_item,
            ]
        )
        result_set = ResultSetEntity(
            model=model,
            ground_truth_dataset=ground_truth_dataset,
            prediction_dataset=prediction_dataset,
        )
        return result_set

    def single_label_result_set(self) -> ResultSetEntity:
        configurable_params = ConfigurableParameters(header="Test model configurable params")
        labels_group = LabelGroup(
            name="single_class_model_labels_group",
            labels=[self.human()],
        )
        model_configuration = ModelConfiguration(configurable_params, LabelSchemaEntity(label_groups=[labels_group]))
        model = ModelEntity(train_dataset=DatasetEntity(), configuration=model_configuration)

        ground_truth_dataset = DatasetEntity(
            [
                self.human_1_dataset_item(),
                self.human_2_dataset_item(),
                self.human_3_dataset_item(),
            ]
        )
        prediction_dataset = DatasetEntity([self.human_1_dataset_item(), self.human_2_dataset_item()])
        result_set = ResultSetEntity(
            model=model,
            ground_truth_dataset=ground_truth_dataset,
            prediction_dataset=prediction_dataset,
        )
        return result_set

    @staticmethod
    def check_confusion_matrix(matrix, expected_name, expected_labels, expected_matrix) -> None:
        assert matrix.name == expected_name
        assert matrix.row_labels == matrix.column_labels
        assert matrix.row_labels == expected_labels
        assert np.array_equal(matrix.matrix_values, expected_matrix)


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestAccuracyFunctions:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_precision_metrics_group(self):
        """
        <b>Description:</b>
        Check "precision_metrics_group" function

        <b>Input data:</b>
        "confusion_matrix" MatrixMetric-class object with specified "name", "matrix_values", "row_labels",
        "column_labels" and "normalize" parameters

        <b>Expected results:</b>
        Test passes if "BarMetricsGroup" object returned by "precision_metrics_group" function is equal to expected

        <b>Steps</b>
        1. Check "BarMetricsGroup" object returned by "precision_metrics_group" function for "confusion_matrix" with
        default value of "row_labels" parameter
        2. Check "BarMetricsGroup" object returned by "precision_metrics_group" function for "confusion_matrix" with
        specified value of "row_labels" parameter
        """

        def check_precision_metrics_group(matrix_for_precision, expected_metrics):
            precision_metrics = precision_metrics_group(matrix_for_precision)
            assert isinstance(precision_metrics, BarMetricsGroup)
            assert precision_metrics.metrics == expected_metrics
            assert isinstance(precision_metrics.visualization_info, BarChartInfo)
            assert precision_metrics.visualization_info.name == "Precision per class"
            assert precision_metrics.visualization_info.palette == ColorPalette.LABEL
            assert precision_metrics.visualization_info.type == VisualizationType.BAR

        matrix_values = np.array([[0, 0.5, 0.5], [0, 0.5, 0.5], [1, 0, 0]])
        # Checking "BarMetricsGroup" object returned by "precision_metrics_group" for "confusion_matrix" with default
        # "row_labels"
        confusion_matrix = MatrixMetric(name="no_row_labels MatrixMetric", matrix_values=matrix_values)
        check_precision_metrics_group(
            matrix_for_precision=confusion_matrix,
            expected_metrics=[
                ScoreMetric(name=np.int32(0), value=0.0),
                ScoreMetric(name=np.int32(1), value=0.5),
                ScoreMetric(name=np.int32(2), value=0.0),
            ],
        )
        # Checking "BarMetricsGroup" object returned by "precision_metrics_group" for "confusion_matrix" with specified
        # "row_labels"
        confusion_matrix = MatrixMetric(
            name="row_labels MatrixMetric",
            matrix_values=matrix_values,
            row_labels=["label for row_1", "label for row_2", "label for row_3"],
            column_labels=[
                "label for column_1",
                "label for column_2",
                "label for column_3",
            ],
        )
        check_precision_metrics_group(
            matrix_for_precision=confusion_matrix,
            expected_metrics=[
                ScoreMetric(name="label for row_1", value=0.0),
                ScoreMetric(name="label for row_2", value=0.5),
                ScoreMetric(name="label for row_3", value=0.0),
            ],
        )

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_recall_metrics_group(self):
        """
        <b>Description:</b>
        Check "recall_metrics_group" function

        <b>Input data:</b>
        MatrixMetric-class object with specified "name", "matrix_values", "row_labels", "column_labels" and "normalize"
        parameters

        <b>Expected results:</b>
        Test passes if "BarMetricsGroup" object returned by "recall_metrics_group" function is equal to expected

        <b>Steps</b>
        1. Check "BarMetricsGroup" object returned by "recall_metrics_group" function when "confusion_matrix"
        initialized with default "row_labels" parameter
        2. Check "BarMetricsGroup" object returned by "recall_metrics_group" function when "confusion_matrix"
        initialized with specified "row_labels" parameter
        """

        def check_recall_metrics_group(matrix_for_recall, expected_metrics):
            recall_metrics = recall_metrics_group(matrix_for_recall)
            assert isinstance(recall_metrics, BarMetricsGroup)
            assert recall_metrics.metrics == expected_metrics
            assert isinstance(recall_metrics.visualization_info, BarChartInfo)
            assert recall_metrics.visualization_info.name == "Recall per class"
            assert recall_metrics.visualization_info.palette == ColorPalette.LABEL
            assert recall_metrics.visualization_info.type == VisualizationType.BAR

        matrix_values = np.array([[0, 0.8, 0.4], [0, 0.8, 0.2], [0.4, 0.2, 0]])
        # Checking "BarMetricsGroup" returned by "recall_metrics_group" when "confusion_matrix" initialized with
        # default "row_labels"
        confusion_matrix = MatrixMetric(name="no_row_labels MatrixMetric", matrix_values=matrix_values)

        check_recall_metrics_group(
            matrix_for_recall=confusion_matrix,
            expected_metrics=[
                ScoreMetric(name=np.int32(0), value=0.0),
                ScoreMetric(name=np.int32(1), value=0.8),
                ScoreMetric(name=np.int32(2), value=0.0),
            ],
        )
        # Checking "BarMetricsGroup" returned by "recall_metrics_group" when "confusion_matrix" initialized with
        # specified "row_labels"
        confusion_matrix = MatrixMetric(
            name="row_labels MatrixMetric",
            matrix_values=matrix_values,
            row_labels=["label for row_1", "label for row_2", "label for row_3"],
            column_labels=[
                "label for column_1",
                "label for column_2",
                "label for column_3",
            ],
        )
        check_recall_metrics_group(
            matrix_for_recall=confusion_matrix,
            expected_metrics=[
                ScoreMetric(name="label for row_1", value=0.0),
                ScoreMetric(name="label for row_2", value=0.8),
                ScoreMetric(name="label for row_3", value=0.0),
            ],
        )

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_compute_unnormalized_confusion_matrices_from_resultset(self):
        """
        <b>Description:</b>
        Check "compute_unnormalized_confusion_matrices_from_resultset" function

        <b>Input data:</b>
        ResultSetEntity-class object with specified "ground_truth_dataset" and "prediction_dataset" parameters

        <b>Expected results:</b>
        Test passes if list returned by "compute_unnormalized_confusion_matrices_from_resultset" function is equal to
        expected

        <b>Steps</b>
        1. Check list returned by "compute_unnormalized_confusion_matrices_from_resultset" function for model with
        multiple labels specified in LabelGroup
        2. Check list returned by "compute_unnormalized_confusion_matrices_from_resultset" function for model with
        single label specified in LabelGroup
        3. Check "ValueError" exception is raised when trying to compute confusion matrix for "ResultSetEntity" object
        with "ground_truth_dataset" attribute equal to empty list
        4. Check "ValueError" exception is raised when trying to compute confusion matrix for "ResultSetEntity" object
        with "prediction_dataset" attribute equal to empty list
        """
        # Checking list returned by "compute_unnormalized_confusion_matrices_from_resultset" for model with multiple
        # labels in LabelGroup
        result_set = CommonActions().result_set()
        confusion_matrices = compute_unnormalized_confusion_matrices_from_resultset(result_set)
        assert len(confusion_matrices) == 2
        # Checking first confusion matrix
        confusion_matrix = confusion_matrices[0]
        CommonActions.check_confusion_matrix(
            matrix=confusion_matrix,
            expected_name="model_labels_group",
            expected_labels=["car", "dog", "human"],
            expected_matrix=np.array([[2, 0, 0], [0, 1, 0], [0, 1, 2]]),
        )
        # Checking second confusion matrix
        confusion_matrix = confusion_matrices[1]
        CommonActions.check_confusion_matrix(
            matrix=confusion_matrix,
            expected_name="other_model_labels_group",
            expected_labels=["dog", "human"],
            expected_matrix=np.array([[1, 0], [1, 2]]),
        )
        # Checking list returned by "compute_unnormalized_confusion_matrices_from_resultset" function for model with
        # single label specified in LabelGroup
        result_set = CommonActions().single_label_result_set()
        confusion_matrices = compute_unnormalized_confusion_matrices_from_resultset(result_set)
        assert len(confusion_matrices) == 1
        confusion_matrix = confusion_matrices[0]
        CommonActions.check_confusion_matrix(
            matrix=confusion_matrix,
            expected_name="single_class_model_labels_group",
            expected_labels=["human", "~ human"],
            expected_matrix=np.array([[2, 0], [0, 0]]),
        )
        # Checking "ValueError" exception is raised when trying to compute confusion matrix for "ResultSetEntity" with
        # "ground_truth_dataset" equal to empty list
        result_set.ground_truth_dataset = []
        with pytest.raises(ValueError):
            compute_unnormalized_confusion_matrices_from_resultset(result_set)
        # Checking "ValueError" exception is raised when trying to compute confusion matrix for "ResultSetEntity" with
        # "prediction_dataset" equal to empty list
        result_set = CommonActions().single_label_result_set()
        result_set.prediction_dataset = []
        with pytest.raises(ValueError):
            compute_unnormalized_confusion_matrices_from_resultset(result_set)


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestAccuracy:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_accuracy_initialization(self):
        """
        <b>Description:</b>
        Check "Accuracy" class object initialization

        <b>Input data:</b>
        "Accuracy" class object with specified "resultset" and "average" parameters

        <b>Expected results:</b>
        Test passes if attributes of initialized "Accuracy" class object are equal to expected

        <b>Steps</b>
        1. Check attributes of "Accuracy" class object initialized with default optional parameters
        2. Check attributes of "Accuracy" class object initialized with specified optional parameters
        """

        def check_confusion_matrices(unnormalized_matrices):
            assert len(unnormalized_matrices) == 2
            # Checking first confusion matrix
            confusion_matrix = unnormalized_matrices[0]
            assert isinstance(confusion_matrix, MatrixMetric)
            CommonActions.check_confusion_matrix(
                matrix=confusion_matrix,
                expected_name="model_labels_group",
                expected_labels=["car", "dog", "human"],
                expected_matrix=np.array([[2, 0, 0], [0, 1, 0], [0, 1, 2]]),
            )
            # Checking second confusion matrix
            confusion_matrix = unnormalized_matrices[1]
            assert isinstance(confusion_matrix, MatrixMetric)
            CommonActions.check_confusion_matrix(
                matrix=confusion_matrix,
                expected_name="other_model_labels_group",
                expected_labels=["dog", "human"],
                expected_matrix=np.array([[1, 0], [1, 2]]),
            )

        result_set = CommonActions().result_set()
        # Checking attributes of "Accuracy" object initialized with default optional parameters
        accuracy = Accuracy(result_set)
        # Checking "accuracy" attribute
        assert accuracy.accuracy == ScoreMetric(name="Accuracy", value=0.8)
        # Checking "unnormalized_matrices" attribute
        actual_unnormalized_matrices = accuracy._unnormalized_matrices
        check_confusion_matrices(actual_unnormalized_matrices)
        # Checking attributes of "Accuracy" object initialized with specified optional parameters
        accuracy = Accuracy(resultset=result_set, average=MetricAverageMethod.MACRO)
        # Checking "accuracy" attribute
        assert accuracy.accuracy == ScoreMetric(name="Accuracy", value=0.7916666666666667)
        # Checking "unnormalized_matrices" attribute
        actual_unnormalized_matrices = accuracy._unnormalized_matrices
        check_confusion_matrices(actual_unnormalized_matrices)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_accuracy_get_performance(self):
        """
        <b>Description:</b>
        Check "Accuracy" class object "get_performance" method

        <b>Input data:</b>
        "Accuracy" class object with specified "resultset" and "average" parameters

        <b>Expected results:</b>
        Test passes if "Performance" object returned by "get_performance" method is equal to expected
        """

        def check_precision_recall_metrics(precision_metric, recall_metric, expected_precision, expected_recall):
            # Checking Precision per class metric
            assert isinstance(precision_metric, BarMetricsGroup)
            assert precision_metric.metrics == expected_precision
            assert precision_metric.visualization_info.name == "Precision per class"
            assert precision_metric.visualization_info.palette == ColorPalette.LABEL
            assert precision_metric.visualization_info.type == VisualizationType.BAR
            # Checking Recall per class metric
            assert isinstance(recall_metric, BarMetricsGroup)
            assert recall_metric.metrics == expected_recall
            assert recall_metric.visualization_info.name == "Recall per class"
            assert recall_metric.visualization_info.palette == ColorPalette.LABEL

        result_set = CommonActions().result_set()
        accuracy = Accuracy(result_set)
        actual_performance = accuracy.get_performance()
        assert isinstance(actual_performance, Performance)
        assert actual_performance.score == ScoreMetric(name="Accuracy", value=0.8)
        assert len(actual_performance.dashboard_metrics) == 5
        # Checking dashboard_metrics
        # Checking first MatrixMetric object
        actual_matrix_metric = actual_performance.dashboard_metrics[0].metrics[0]
        CommonActions.check_confusion_matrix(
            matrix=actual_matrix_metric,
            expected_name="model_labels_group",
            expected_labels=["car", "dog", "human"],
            expected_matrix=np.array([[1, 0, 0], [0, 1, 0], [0, 0.33333334, 0.6666667]], dtype=np.float32),
        )
        # Checking second MatrixMetric object
        actual_matrix_metric = actual_performance.dashboard_metrics[0].metrics[1]
        CommonActions.check_confusion_matrix(
            matrix=actual_matrix_metric,
            expected_name="other_model_labels_group",
            expected_labels=["dog", "human"],
            expected_matrix=np.array([[1, 0], [0.33333334, 0.6666667]], dtype=np.float32),
        )
        # Checking Precision and Recall BarMetricsGroup for first label_group
        precision_per_class_metric = actual_performance.dashboard_metrics[1]
        recall_per_class_metric = actual_performance.dashboard_metrics[2]
        check_precision_recall_metrics(
            precision_metric=precision_per_class_metric,
            recall_metric=recall_per_class_metric,
            expected_precision=[
                ScoreMetric("car", 1.0),
                ScoreMetric("dog", 0.5),
                ScoreMetric("human", 1.0),
            ],
            expected_recall=[
                ScoreMetric("car", 1.0),
                ScoreMetric("dog", 1.0),
                ScoreMetric("human", 0.6666666666666666),
            ],
        )
        # Checking Precision and Recall BarMetricsGroup for second label_group
        precision_per_class_metric = actual_performance.dashboard_metrics[3]
        recall_per_class_metric = actual_performance.dashboard_metrics[4]
        check_precision_recall_metrics(
            precision_metric=precision_per_class_metric,
            recall_metric=recall_per_class_metric,
            expected_precision=[ScoreMetric("dog", 0.5), ScoreMetric("human", 1.0)],
            expected_recall=[
                ScoreMetric("dog", 1.0),
                ScoreMetric("human", 0.6666666666666666),
            ],
        )

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_accuracy_compute_accuracy(self):
        """
        <b>Description:</b>
        Check "Accuracy" class object "_compute_accuracy" method

        <b>Input data:</b>
        "confusion_matrices" list and "average" MetricAverageMethod parameters

        <b>Expected results:</b>
        Test passes if value returned by "_compute_accuracy" method is equal to expected

        <b>Steps</b>
        1. Check value returned by "_compute_accuracy" method for single confusion matrix specified as
        "confusion_matrices" parameter
        2. Check value returned by "_compute_accuracy" method for several confusion matrices specified as
        "confusion_matrices" parameter
        3. Check "ValueError" exception is raised when empty list is specified as "confusion_matrices" parameter
        4. Check "RuntimeError" exception is raised when unexpected method is specified as "average" parameter
        """
        result_set = CommonActions().single_label_result_set()
        accuracy = Accuracy(result_set)
        # Checking value returned by "_compute_accuracy" for single confusion matrix specified as "confusion_matrices"
        confusion_matrix = MatrixMetric(
            name="confusion_matrix",
            matrix_values=np.array([[6, 1, 0], [2, 6, 1], [0, 0, 4]]),
        )
        expected_accuracy = np.float64(0.8)
        assert (
            accuracy._compute_accuracy(average=MetricAverageMethod.MICRO, confusion_matrices=[confusion_matrix])
            == expected_accuracy
        )
        assert (
            accuracy._compute_accuracy(average=MetricAverageMethod.MACRO, confusion_matrices=[confusion_matrix])
            == expected_accuracy
        )
        # Checking value returned by "_compute_accuracy" for several confusion matrices specified as
        # "confusion_matrices"
        other_confusion_matrix = MatrixMetric(
            name="other_confusion_matrix",
            matrix_values=np.array([[4, 0, 0], [2, 4, 0], [0, 0, 6]]),
        )
        assert accuracy._compute_accuracy(
            average=MetricAverageMethod.MICRO,
            confusion_matrices=[confusion_matrix, other_confusion_matrix],
        ) == np.float64(0.8333333333333334)
        assert accuracy._compute_accuracy(
            average=MetricAverageMethod.MACRO,
            confusion_matrices=[confusion_matrix, other_confusion_matrix],
        ) == np.float64(0.8375)
        # Checking "ValueError" exception is raised when empty list is specified as "confusion_matrices"
        with pytest.raises(ValueError):
            accuracy._compute_accuracy(average=MetricAverageMethod.MACRO, confusion_matrices=[])
        # Checking "RuntimeError" exception is raised when unexpected method is specified as "average"
        with pytest.raises(RuntimeError):
            accuracy._compute_accuracy(average="unknown average", confusion_matrices=[confusion_matrix])
