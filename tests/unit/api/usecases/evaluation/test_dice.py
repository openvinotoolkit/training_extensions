# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import datetime

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
    Performance,
    ScoreMetric,
    VisualizationType,
)
from otx.api.entities.model import ModelConfiguration, ModelEntity
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.usecases.evaluation.averaging import MetricAverageMethod
from otx.api.usecases.evaluation.dice import DiceAverage
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestDice:
    color = Color(0, 255, 0)
    creation_date = datetime.datetime(year=2022, month=1, day=10)
    image = Image(data=np.random.randint(low=0, high=255, size=(64, 32, 3)))
    full_box_roi = Rectangle.generate_full_box()
    car_label = LabelEntity(
        name="car",
        domain=Domain.DETECTION,
        color=color,
        creation_date=creation_date,
        id=ID("car_label"),
    )
    human_label = LabelEntity(
        name="human",
        domain=Domain.DETECTION,
        color=color,
        creation_date=creation_date,
        id=ID("human_label"),
    )
    dog_label = LabelEntity(
        name="dog",
        domain=Domain.DETECTION,
        color=color,
        creation_date=creation_date,
        id=ID("dog_label"),
    )
    cat_label = LabelEntity(
        name="cat",
        domain=Domain.DETECTION,
        color=color,
        creation_date=creation_date,
        id=ID("cat_label"),
    )
    configurable_params = ConfigurableParameters(header="Test model configurable params")

    def human_1_ground_truth(self) -> DatasetItemEntity:
        human_roi = Annotation(shape=self.full_box_roi, labels=[ScoredLabel(self.human_label)])
        human_annotation = Annotation(
            shape=Rectangle(x1=0.4, y1=0, x2=0.5, y2=0.2),
            labels=[ScoredLabel(self.human_label)],
        )
        human_annotation_scene = AnnotationSceneEntity(
            annotations=[human_annotation], kind=AnnotationSceneKind.ANNOTATION
        )
        human_dataset_item = DatasetItemEntity(media=self.image, annotation_scene=human_annotation_scene, roi=human_roi)
        return human_dataset_item

    def human_2_ground_truth(self) -> DatasetItemEntity:
        human_roi = Annotation(shape=self.full_box_roi, labels=[ScoredLabel(self.human_label)])
        human_annotation = Annotation(
            shape=Rectangle(x1=0.6, y1=0, x2=0.7, y2=0.2),
            labels=[ScoredLabel(self.human_label)],
        )
        human_annotation_scene = AnnotationSceneEntity(
            annotations=[human_annotation], kind=AnnotationSceneKind.ANNOTATION
        )
        human_dataset_item = DatasetItemEntity(media=self.image, annotation_scene=human_annotation_scene, roi=human_roi)
        return human_dataset_item

    def dog_ground_truth(self) -> DatasetItemEntity:
        dog_roi = Annotation(shape=self.full_box_roi, labels=[ScoredLabel(self.dog_label)])
        dog_annotation = Annotation(
            shape=Rectangle(x1=0.8, y1=0, x2=0.9, y2=0.1),
            labels=[ScoredLabel(self.dog_label)],
        )
        dog_annotation_scene = AnnotationSceneEntity(annotations=[dog_annotation], kind=AnnotationSceneKind.ANNOTATION)
        dog_dataset_item = DatasetItemEntity(media=self.image, annotation_scene=dog_annotation_scene, roi=dog_roi)
        return dog_dataset_item

    def human_1_predicted(self) -> DatasetItemEntity:
        human_roi = Annotation(shape=self.full_box_roi, labels=[ScoredLabel(self.human_label)])
        human_annotation = Annotation(
            shape=Rectangle(x1=0.4, y1=0, x2=0.5, y2=0.4),
            labels=[ScoredLabel(self.human_label)],
        )
        human_annotation_scene = AnnotationSceneEntity(
            annotations=[human_annotation], kind=AnnotationSceneKind.ANNOTATION
        )
        human_dataset_item = DatasetItemEntity(media=self.image, annotation_scene=human_annotation_scene, roi=human_roi)
        return human_dataset_item

    def human_2_predicted(self) -> DatasetItemEntity:
        human_roi = Annotation(shape=self.full_box_roi, labels=[ScoredLabel(self.human_label)])
        human_annotation = Annotation(
            shape=Rectangle(x1=0.6, y1=0, x2=0.7, y2=0.2),
            labels=[ScoredLabel(self.human_label)],
        )
        human_annotation_scene = AnnotationSceneEntity(
            annotations=[human_annotation], kind=AnnotationSceneKind.ANNOTATION
        )
        human_dataset_item = DatasetItemEntity(media=self.image, annotation_scene=human_annotation_scene, roi=human_roi)
        return human_dataset_item

    def cat_predicted(self) -> DatasetItemEntity:
        cat_roi = Annotation(shape=self.full_box_roi, labels=[ScoredLabel(self.cat_label)])
        cat_annotation = Annotation(
            shape=Rectangle(x1=0.9, y1=0, x2=1.0, y2=0.1),
            labels=[ScoredLabel(self.cat_label)],
        )
        cat_annotation_scene = AnnotationSceneEntity(annotations=[cat_annotation], kind=AnnotationSceneKind.ANNOTATION)
        cat_dataset_item = DatasetItemEntity(media=self.image, annotation_scene=cat_annotation_scene, roi=cat_roi)
        return cat_dataset_item

    def car_dataset_item(self) -> DatasetItemEntity:
        car_roi = Annotation(
            shape=self.full_box_roi,
            labels=[ScoredLabel(self.car_label)],
        )
        car_annotation = Annotation(
            shape=Rectangle(x1=0.1, y1=0, x2=0.3, y2=0.2),
            labels=[ScoredLabel(self.car_label)],
        )
        car_annotation_scene = AnnotationSceneEntity(annotations=[car_annotation], kind=AnnotationSceneKind.ANNOTATION)
        car_dataset_item = DatasetItemEntity(media=self.image, annotation_scene=car_annotation_scene, roi=car_roi)
        return car_dataset_item

    def model(self) -> ModelEntity:
        labels_group = LabelGroup(
            name="model_labels_group",
            labels=[self.car_label, self.human_label, self.dog_label, self.cat_label],
        )
        model_configuration = ModelConfiguration(
            configurable_parameters=self.configurable_params,
            label_schema=LabelSchemaEntity(label_groups=[labels_group]),
        )
        model = ModelEntity(train_dataset=DatasetEntity(), configuration=model_configuration)
        return model

    @staticmethod
    def check_score_metric(score_metric, expected_name, expected_value):
        assert isinstance(score_metric, ScoreMetric)
        assert score_metric.name == expected_name
        assert score_metric.value == pytest.approx(expected_value)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dice_initialization(self):
        """
        <b>Description:</b>
        Check "DiceAverage" class object initialization

        <b>Input data:</b>
        "DiceAverage" object with specified "resultset" and "average" parameters

        <b>Expected results:</b>
        Test passes if attributes of initialized "DiceAverage" object are equal to expected

        <b>Steps</b>
        1. Check attributes of "DiceAverage" object initialized with default value of "average" parameter
        2. Check attributes of "DiceAverage" object initialized with specified value of "average" parameter
        3. Check that "ValueError" exception is raised when initializing "DiceAverage" object with empty list prediction
        "resultset" attribute
        4. Check "ValueError" exception is raised when initializing "DiceAverage" object with "resultset" attribute with
        unequal length of "ground_truth_dataset" and "prediction_dataset"
        """

        def check_dice_attributes(
            dice_actual: DiceAverage,
            expected_average_type: MetricAverageMethod,
            expected_overall_dice: float,
        ):
            assert dice_actual.average == expected_average_type
            # Checking "overall_dice" attribute
            self.check_score_metric(
                score_metric=dice_actual.overall_dice,
                expected_name="Dice Average",
                expected_value=expected_overall_dice,
            )
            # Checking "dice_per_label" attribute
            assert len(dice_actual.dice_per_label) == 4
            self.check_score_metric(
                score_metric=dice_actual.dice_per_label.get(self.car_label),
                expected_name="car",
                expected_value=1.0,
            )
            self.check_score_metric(
                score_metric=dice_actual.dice_per_label.get(self.human_label),
                expected_name="human",
                expected_value=0.782608695652174,
            )
            self.check_score_metric(
                score_metric=dice_actual.dice_per_label.get(self.dog_label),
                expected_name="dog",
                expected_value=0.0,
            )
            self.check_score_metric(
                score_metric=dice_actual.dice_per_label.get(self.cat_label),
                expected_name="cat",
                expected_value=0.0,
            )

        model = self.model()
        human_1_ground_truth = self.human_1_ground_truth()
        car_dataset_item = self.car_dataset_item()
        ground_truth_dataset = DatasetEntity(
            [
                car_dataset_item,
                human_1_ground_truth,
                self.human_2_ground_truth(),
                self.dog_ground_truth(),
            ]
        )
        predicted_dataset = DatasetEntity(
            [
                car_dataset_item,
                self.human_1_predicted(),
                self.human_2_predicted(),
                self.cat_predicted(),
            ]
        )
        result_set = ResultSetEntity(
            model=model,
            ground_truth_dataset=ground_truth_dataset,
            prediction_dataset=predicted_dataset,
        )
        # Checking attributes of "DiceAverage" initialized with default "average"
        dice = DiceAverage(resultset=result_set)
        check_dice_attributes(
            dice_actual=dice,
            expected_average_type=MetricAverageMethod.MACRO,
            expected_overall_dice=0.44565217391304346,
        )
        # Checking attributes of "DiceAverage" initialized with specified "average"
        dice = DiceAverage(resultset=result_set, average=MetricAverageMethod.MICRO)
        check_dice_attributes(
            dice_actual=dice,
            expected_average_type=MetricAverageMethod.MICRO,
            expected_overall_dice=0.7746741154562383,
        )
        # Checking "ValueError" exception raised when initializing "DiceAverage" with empty list prediction result_set
        result_set = ResultSetEntity(
            model=model,
            ground_truth_dataset=ground_truth_dataset,
            prediction_dataset=DatasetEntity(items=[]),
        )
        with pytest.raises(ValueError):
            DiceAverage(resultset=result_set)

        # Checking "ValueError" exception raised when initializing "DiceAverage" with "resultset" with unequal length of
        # "ground_truth_dataset" and "prediction_dataset"
        result_set = ResultSetEntity(
            model=model,
            ground_truth_dataset=DatasetEntity([car_dataset_item, human_1_ground_truth]),
            prediction_dataset=DatasetEntity([car_dataset_item]),
        )
        with pytest.raises(ValueError):
            DiceAverage(resultset=result_set)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dice_get_performance(self):
        """
        <b>Description:</b>
        Check "DiceAverage" class "get_performance" method

        <b>Input data:</b>
        "DiceAverage" object with specified "resultset" and "average" parameters

        <b>Expected results:</b>
        Test passes if "Performance" object returned by "get_performance" method is equal to expected

        <b>Steps</b>
        1. Check "Performance" object returned by "get_performance" method for "DiceAverage" class object with length of
        "dice_per_label" attribute more than 0
        2. Check "Performance" object returned by "get_performance" method for "DiceAverage" class object with length of
        "dice_per_label" attribute equal to 0
        """
        human_1_ground_truth = self.human_1_ground_truth()
        human_1_predicted = self.human_1_predicted()
        car_dataset_item = self.car_dataset_item()
        # Checking "Performance" returned by "get_performance" for "DiceAverage" with length of "dice_per_label" more
        # than 0
        configurable_params = self.configurable_params
        labels_group = LabelGroup(
            name="model_labels_group",
            labels=[self.car_label, self.human_label, self.dog_label, self.cat_label],
        )
        model_configuration = ModelConfiguration(
            configurable_parameters=configurable_params,
            label_schema=LabelSchemaEntity(label_groups=[labels_group]),
        )
        model = ModelEntity(train_dataset=DatasetEntity(), configuration=model_configuration)
        ground_truth_dataset = DatasetEntity(
            [
                car_dataset_item,
                human_1_ground_truth,
                self.human_2_ground_truth(),
                self.dog_ground_truth(),
            ]
        )
        predicted_dataset = DatasetEntity(
            [
                car_dataset_item,
                human_1_predicted,
                self.human_2_predicted(),
                self.cat_predicted(),
            ]
        )
        result_set = ResultSetEntity(
            model=model,
            ground_truth_dataset=ground_truth_dataset,
            prediction_dataset=predicted_dataset,
        )
        dice = DiceAverage(resultset=result_set)
        performance = dice.get_performance()
        assert isinstance(performance, Performance)
        # Checking "score" attribute
        self.check_score_metric(
            score_metric=performance.score,
            expected_name="Dice Average",
            expected_value=0.44565217391304346,
        )
        # Checking "dashboard_metrics" attribute
        assert len(performance.dashboard_metrics) == 1
        dashboard_metric = performance.dashboard_metrics[0]
        assert isinstance(dashboard_metric, BarMetricsGroup)
        # Checking "metrics" attribute
        assert len(dashboard_metric.metrics) == 4
        self.check_score_metric(
            score_metric=dashboard_metric.metrics[0],
            expected_name="car",
            expected_value=1.0,
        )
        self.check_score_metric(
            score_metric=dashboard_metric.metrics[1],
            expected_name="cat",
            expected_value=0.0,
        )
        self.check_score_metric(
            score_metric=dashboard_metric.metrics[2],
            expected_name="dog",
            expected_value=0.0,
        )
        self.check_score_metric(
            score_metric=dashboard_metric.metrics[3],
            expected_name="human",
            expected_value=0.782608695652174,
        )
        # Checking "visualization_info" attribute
        assert isinstance(dashboard_metric.visualization_info, BarChartInfo)
        assert dashboard_metric.visualization_info.name == "Dice Average Per Label"
        assert dashboard_metric.visualization_info.palette == ColorPalette.LABEL
        assert dashboard_metric.visualization_info.type == VisualizationType.BAR
        # Checking "Performance" returned by "get_performance" for "DiceAverage" with length of "dice_per_label" equal
        # to 0
        labels_group = LabelGroup(name="model_labels_group", labels=[self.car_label])
        model_configuration = ModelConfiguration(configurable_params, LabelSchemaEntity(label_groups=[labels_group]))
        model = ModelEntity(train_dataset=DatasetEntity(), configuration=model_configuration)
        ground_truth_dataset = DatasetEntity([human_1_ground_truth])
        predicted_dataset = DatasetEntity([human_1_predicted])
        result_set = ResultSetEntity(
            model=model,
            ground_truth_dataset=ground_truth_dataset,
            prediction_dataset=predicted_dataset,
        )
        dice = DiceAverage(resultset=result_set)
        performance = dice.get_performance()
        assert isinstance(performance, Performance)
        # Checking "score" attribute
        self.check_score_metric(
            score_metric=performance.score,
            expected_name="Dice Average",
            expected_value=0.0,
        )
        # Checking "dashboard_metrics" attribute
        assert performance.dashboard_metrics == []

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dice_compute_dice_using_intersection_and_cardinality(self):
        """
        <b>Description:</b>
        Check "DiceAverage" class "compute_dice_using_intersection_and_cardinality" method

        <b>Input data:</b>
        "DiceAverage" object, "all_intersection" dictionary, "all_cardinality" dictionary, "average" MetricAverageMethod
        object

        <b>Expected results:</b>
        Test passes if tuple returned by "compute_dice_using_intersection_and_cardinality" method is equal to expected

        <b>Steps</b>
        1. Check tuple returned by "compute_dice_using_intersection_and_cardinality" method for MICRO "average"
        parameter
        2. Check tuple returned by "compute_dice_using_intersection_and_cardinality" method for MACRO "average"
        parameter
        3. Check tuple returned by "compute_dice_using_intersection_and_cardinality" method when "None" key is not
        specified in "all_intersection" and "all_cardinality" dictionaries
        4. Check "KeyError" exception is raised when keys of "all_intersection" and "all_cardinality" dictionaries are
        not match
        5. Check "ValueError" exception is raised when intersection of certain key is larger than its cardinality
        """

        def check_dice(dice_actual: tuple, expected_overall_dice: float):
            assert len(dice_actual) == 2
            self.check_score_metric(
                score_metric=dice_actual[0],
                expected_name="Dice Average",
                expected_value=expected_overall_dice,
            )
            dice_per_label = dice_actual[1]
            assert len(dice_per_label) == 4
            self.check_score_metric(
                score_metric=dice_per_label.get(self.car_label),
                expected_name="car",
                expected_value=2.0,
            )
            self.check_score_metric(
                score_metric=dice_per_label.get(self.human_label),
                expected_name="human",
                expected_value=1.6,
            )
            self.check_score_metric(
                score_metric=dice_per_label.get(self.dog_label),
                expected_name="dog",
                expected_value=0.5,
            )
            self.check_score_metric(
                score_metric=dice_per_label.get(self.cat_label),
                expected_name="cat",
                expected_value=0.0,
            )

        # Checking tuple returned by "compute_dice_using_intersection_and_cardinality" for MICRO "average"
        all_intersection = {
            self.car_label: 10,
            self.human_label: 8,
            self.dog_label: 2,
            self.cat_label: 0,
            None: 9,
        }
        all_cardinality = {
            self.car_label: 10,
            self.human_label: 10,
            self.dog_label: 8,
            self.cat_label: 2,
            None: 12,
        }
        dice = DiceAverage.compute_dice_using_intersection_and_cardinality(
            all_intersection=all_intersection,
            all_cardinality=all_cardinality,
            average=MetricAverageMethod.MICRO,
        )
        check_dice(dice_actual=dice, expected_overall_dice=1.5)
        # Checking tuple returned by "compute_dice_using_intersection_and_cardinality" for MACRO "average"
        dice = DiceAverage.compute_dice_using_intersection_and_cardinality(
            all_intersection=all_intersection,
            all_cardinality=all_cardinality,
            average=MetricAverageMethod.MACRO,
        )
        check_dice(dice_actual=dice, expected_overall_dice=1.025)
        # Checking tuple returned by "compute_dice_using_intersection_and_cardinality" when "None" key is not
        # specified in "all_intersection" and "all_cardinality" dictionaries
        all_intersection = {
            self.car_label: 10,
            self.human_label: 8,
            self.dog_label: 2,
            self.cat_label: 0,
        }
        all_cardinality = {
            self.car_label: 10,
            self.human_label: 10,
            self.dog_label: 8,
            self.cat_label: 2,
        }
        # Check for MACRO "average" parameter
        dice = DiceAverage.compute_dice_using_intersection_and_cardinality(
            all_intersection=all_intersection,
            all_cardinality=all_cardinality,
            average=MetricAverageMethod.MACRO,
        )
        check_dice(dice_actual=dice, expected_overall_dice=1.025)
        # Expected KeyError exception for MICRO "average" parameter
        with pytest.raises(KeyError):
            DiceAverage.compute_dice_using_intersection_and_cardinality(
                all_intersection=all_intersection,
                all_cardinality=all_cardinality,
                average=MetricAverageMethod.MICRO,
            )
        # Checking "KeyError" exception is raised when keys of "all_intersection" and "all_cardinality" dictionaries are
        # not match
        all_intersection = {self.car_label: 10, self.human_label: 9, None: 9}
        all_cardinality = {
            self.car_label: 10,
            self.dog_label: 8,
            self.cat_label: 2,
            None: 12,
        }
        with pytest.raises(KeyError):
            DiceAverage.compute_dice_using_intersection_and_cardinality(
                all_intersection=all_intersection,
                all_cardinality=all_cardinality,
                average=MetricAverageMethod.MACRO,
            )
        # Checking "ValueError" exception is raised when intersection of certain key is larger than its cardinality
        all_intersection = {self.car_label: 10, self.human_label: 9, None: 12}
        all_cardinality = {self.car_label: 10, self.human_label: 8, None: 12}
        with pytest.raises(ValueError):
            DiceAverage.compute_dice_using_intersection_and_cardinality(
                all_intersection=all_intersection,
                all_cardinality=all_cardinality,
                average=MetricAverageMethod.MACRO,
            )
