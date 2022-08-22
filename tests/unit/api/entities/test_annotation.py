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

import datetime
from typing import List

import pytest

from otx.api.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
    NullAnnotationSceneEntity,
)
from otx.api.entities.color import Color
from otx.api.entities.id import ID
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.ellipse import Ellipse
from otx.api.entities.shapes.polygon import Point, Polygon
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.utils.time_utils import now
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestAnnotation:

    rectangle = Rectangle(x1=0.5, x2=1.0, y1=0.0, y2=0.5)
    labels: List[ScoredLabel] = []
    annotation = Annotation(shape=rectangle, labels=labels)

    car = LabelEntity(
        id=ID(123456789),
        name="car",
        domain=Domain.DETECTION,
        color=Color(red=16, green=15, blue=56, alpha=255),
        is_empty=True,
    )
    person = LabelEntity(
        id=ID(987654321),
        name="person",
        domain=Domain.DETECTION,
        color=Color(red=11, green=18, blue=38, alpha=200),
        is_empty=False,
    )
    car_label = ScoredLabel(car)
    person_label = ScoredLabel(person)
    labels2 = [car_label, person_label]

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_annotation_default_property(self):
        """
        <b>Description:</b>
        Check that Annotation can correctly return default property value

        <b>Input data:</b>
        Annotation class

        <b>Expected results:</b>
        Test passes if the Annotation return correct values

        <b>Steps</b>
        1. Create Annotation instances
        2. Check default values
        """

        annotation = self.annotation

        assert type(annotation.id_) == ID
        assert annotation.id_ is not None
        assert str(annotation.shape) == "Rectangle(x=0.5, y=0.0, width=0.5, height=0.5)"
        assert annotation.get_labels() == []

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_annotation_setters(self):
        """
        <b>Description:</b>
        Check that Annotation can correctly return modified property value

        <b>Input data:</b>
        Annotation class

        <b>Expected results:</b>
        Test passes if the Annotation return correct values

        <b>Steps</b>
        1. Create Annotation instances
        2. Set another values
        3. Check changed values
        """

        annotation = self.annotation
        ellipse = Ellipse(x1=0.5, y1=0.1, x2=0.8, y2=0.3)
        annotation.shape = ellipse
        annotation.id_ = ID(123456789)

        assert annotation.id_ == ID(123456789)
        assert annotation.shape == ellipse

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_annotation_magic_methods(self):
        """
        <b>Description:</b>
        Check Annotation __repr__, __eq__ methods

        <b>Input data:</b>
        Initialized instance of Annotation

        <b>Expected results:</b>
        Test passes if Annotation magic methods returns correct values

        <b>Steps</b>
        1. Create Annotation instances
        2. Check returning value of magic methods
        """

        annotation = self.annotation
        other_annotation = self.annotation

        point1 = Point(0.3, 0.1)
        point2 = Point(0.8, 0.3)
        point3 = Point(0.6, 0.2)
        points = [point1, point2, point3]
        third_annotation = Annotation(shape=Polygon(points=points), labels=self.labels)

        assert repr(annotation) == "Annotation(shape=Ellipse(x1=0.5, y1=0.1, x2=0.8, y2=0.3), labels=[], id=123456789)"
        assert annotation == other_annotation
        assert annotation != third_annotation
        assert annotation != str

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_annotation_get_labels(self):
        """
        <b>Description:</b>
        Check Annotation get_labels method

        <b>Input data:</b>
        Initialized instance of Annotation

        <b>Expected results:</b>
        Test passes if Annotation get_labels method returns correct values

        <b>Steps</b>
        1. Create Annotation instances
        2. Check returning value of get_labels method
        3. Check returning value of get_labels method with include_empty=True
        """
        annotation = Annotation(shape=self.rectangle, labels=self.labels2)

        assert "[ScoredLabel(987654321, name=person, probability=0.0, domain=DETECTION," in str(annotation.get_labels())
        assert "color=Color(red=11, green=18, blue=38, alpha=200), hotkey=" in str(annotation.get_labels())
        assert ", label_source=LabelSource(user_id='', model_id=ID(), model_storage_id=ID()))]" in str(
            annotation.get_labels()
        )

        assert "[ScoredLabel(123456789, name=car" in str(annotation.get_labels(include_empty=True))
        assert ", probability=0.0, domain=DETECTION," in str(annotation.get_labels(include_empty=True))
        assert "color=Color(red=16, green=15," in str(annotation.get_labels(include_empty=True))
        assert "blue=56, alpha=255), hotkey=," in str(annotation.get_labels(include_empty=True))
        assert "label_source=LabelSource(user_id='', model_id=ID(), model_storage_id=ID()))," in str(
            annotation.get_labels(include_empty=True)
        )

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_annotation_get_label_ids(self):
        """
        <b>Description:</b>
        Check Annotation get_label_ids method

        <b>Input data:</b>
        Initialized instance of Annotation

        <b>Expected results:</b>
        Test passes if Annotation get_label_ids method returns correct values

        <b>Steps</b>
        1. Create Annotation instances
        2. Check returning value of get_label_ids method
        3. Check returning value of get_label_ids method with include_empty=True
        """

        annotation = Annotation(shape=self.rectangle, labels=self.labels2)

        assert annotation.get_label_ids() == {ID(987654321)}
        assert annotation.get_label_ids(include_empty=True) == {
            ID(987654321),
            ID(123456789),
        }

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_annotation_append_label(self):
        """
        <b>Description:</b>
        Check Annotation append_label method

        <b>Input data:</b>
        Initialized instance of Annotation

        <b>Expected results:</b>
        Test passes if Annotation append_label method correct appending label

        <b>Steps</b>
        1. Create Annotation instances
        2. Append label
        3. Check labels
        """

        annotation = self.annotation

        annotation.append_label(label=self.car_label)
        assert annotation.get_labels() == []  # car_label is empty

        annotation.append_label(label=self.person_label)
        assert "name=person" in str(annotation.get_labels())

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_annotation_set_labels(self):
        """
        <b>Description:</b>
        Check Annotation set_labels method

        <b>Input data:</b>
        Initialized instance of Annotation

        <b>Expected results:</b>
        Test passes if Annotation set_labels method correct setting label

        <b>Steps</b>
        1. Create Annotation instances
        2. Set labels
        3. Check labels
        """

        annotation = self.annotation
        assert annotation.get_labels() != []

        annotation.set_labels(labels=[])
        assert annotation.get_labels() == []

        annotation.set_labels(labels=self.labels2)
        assert "name=person" in str(annotation.get_labels())
        assert "name=car" not in str(annotation.get_labels())  # car_label is empty


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestAnnotationSceneKind:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_annotation_scene_kind(self):
        """
        <b>Description:</b>
        Check that AnnotationSceneKind Enum lenght

        <b>Input data:</b>
        AnnotationSceneKind class

        <b>Expected results:</b>
        Test passes if the AnnotationSceneKind return correct values

        <b>Steps</b>
        1. Create AnnotationSceneKind instances
        2. Check Enum lenght
        """

        annotation_scene_kind = AnnotationSceneKind
        assert len(annotation_scene_kind) == 6

        assert str(annotation_scene_kind(0)) == "NONE"
        assert str(annotation_scene_kind(1)) == "ANNOTATION"
        assert str(annotation_scene_kind(2)) == "PREDICTION"
        assert str(annotation_scene_kind(3)) == "EVALUATION"
        assert str(annotation_scene_kind(4)) == "INTERMEDIATE"
        assert str(annotation_scene_kind(5)) == "TASK_PREDICTION"


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestAnnotationSceneEntity:

    creation_date = now()
    labels: List[ScoredLabel] = []
    rectangle = Rectangle(x1=0.5, x2=1.0, y1=0.0, y2=0.5)
    annotation = Annotation(shape=rectangle, labels=labels)

    point1 = Point(0.3, 0.1)
    point2 = Point(0.8, 0.3)
    point3 = Point(0.6, 0.2)
    points = [point1, point2, point3]
    polygon = Polygon(points=points)
    annotation2 = Annotation(shape=polygon, labels=labels)

    annotations = [annotation, annotation2]

    annotation_scene_entity = AnnotationSceneEntity(annotations=annotations, kind=AnnotationSceneKind.ANNOTATION)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_annotation_scene_entity_default_value(self):
        """
        <b>Description:</b>
        Check that AnnotationSceneEntity default values

        <b>Input data:</b>
        AnnotationSceneEntity class

        <b>Expected results:</b>
        Test passes if the AnnotationSceneEntity return correct values

        <b>Steps</b>
        1. Create AnnotationSceneEntity instances
        2. Check default values
        """

        annotation_scene_entity = self.annotation_scene_entity

        assert annotation_scene_entity.id_ == ID()
        assert annotation_scene_entity.kind == AnnotationSceneKind.ANNOTATION
        assert annotation_scene_entity.editor_name == ""
        assert type(annotation_scene_entity.creation_date) == datetime.datetime
        assert "Annotation(shape=Rectangle" in str(annotation_scene_entity.annotations)
        assert "Annotation(shape=Polygon" in str(annotation_scene_entity.annotations)
        assert annotation_scene_entity.shapes == [self.rectangle, self.polygon]

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_annotation_scene_entity_setters(self):
        """
        <b>Description:</b>
        Check that AnnotationSceneEntity can correctly return modified property value

        <b>Input data:</b>
        Annotation class

        <b>Expected results:</b>
        Test passes if the AnnotationSceneEntity return correct values

        <b>Steps</b>
        1. Create AnnotationSceneEntity instances
        2. Set another values
        3. Check changed values
        """

        annotation_scene_entity = self.annotation_scene_entity

        creation_date = self.creation_date
        annotation_scene_entity.id_ = ID(123456789)
        annotation_scene_entity.kind = AnnotationSceneKind.PREDICTION
        annotation_scene_entity.editor_name = "editor"
        annotation_scene_entity.creation_date = creation_date
        annotation_scene_entity.annotations = self.annotation

        assert annotation_scene_entity.id_ == ID(123456789)
        assert annotation_scene_entity.kind == AnnotationSceneKind.PREDICTION
        assert annotation_scene_entity.editor_name == "editor"
        assert annotation_scene_entity.creation_date == creation_date
        assert annotation_scene_entity.annotations == self.annotation

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_annotation_scene_entity_magic_methods(self):
        """
        <b>Description:</b>
        Check Annotation __repr__ method

        <b>Input data:</b>
        Initialized instance of AnnotationSceneEntity

        <b>Expected results:</b>
        Test passes if AnnotationSceneEntity magic method returns correct values

        <b>Steps</b>
        1. Create AnnotationSceneEntity instances
        2. Check returning value of magic method
        """

        annotation_scene_entity = self.annotation_scene_entity

        annotation_scene_entity_repr = [
            f"{annotation_scene_entity.__class__.__name__}("
            f"annotations={annotation_scene_entity.annotations}, "
            f"kind={annotation_scene_entity.kind}, "
            f"editor={annotation_scene_entity.editor_name}, "
            f"creation_date={annotation_scene_entity.creation_date}, "
            f"id={annotation_scene_entity.id_})"
        ]

        for i in annotation_scene_entity_repr:
            assert i in repr(annotation_scene_entity)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_annotation_scene_entity_contains_any(self):
        """
        <b>Description:</b>
        Check Annotation contains_any method

        <b>Input data:</b>
        Initialized instance of AnnotationSceneEntity

        <b>Expected results:</b>
        Test passes if AnnotationSceneEntity contains_any method returns correct values

        <b>Steps</b>
        1. Create AnnotationSceneEntity instances
        2. Check returning value of contains_any method
        """

        annotation_scene_entity = self.annotation_scene_entity
        annotation_scene_entity.annotations = self.annotations

        car = LabelEntity(name="car", domain=Domain.DETECTION, is_empty=True)
        person = LabelEntity(name="person", domain=Domain.DETECTION)
        tree = LabelEntity(name="tree", domain=Domain.DETECTION)
        car_label = ScoredLabel(car)
        person_label = ScoredLabel(person)
        tree_label = ScoredLabel(tree)
        labels = [car_label]
        labels2 = [car_label, person_label]

        annotation = Annotation(shape=self.rectangle, labels=labels2)
        annotations = [annotation]
        annotation_scene_entity2 = AnnotationSceneEntity(annotations=annotations, kind=AnnotationSceneKind.ANNOTATION)

        assert annotation_scene_entity.contains_any(labels=labels) is False
        assert annotation_scene_entity2.contains_any(labels=labels2) is True
        assert annotation_scene_entity2.contains_any(labels=[tree_label]) is False

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_annotation_scene_entity_append_annotation(self):
        """
        <b>Description:</b>
        Check Annotation append_annotation method

        <b>Input data:</b>
        Initialized instance of AnnotationSceneEntity

        <b>Expected results:</b>
        Test passes if AnnotationSceneEntity append_annotation method returns correct values

        <b>Steps</b>
        1. Create AnnotationSceneEntity instances
        2. Check returning value of append_annotation method
        """

        annotation_scene_entity = self.annotation_scene_entity

        tree = LabelEntity(name="tree", domain=Domain.DETECTION)
        tree_label = ScoredLabel(tree)
        labels = [tree_label]
        annotation = Annotation(shape=self.rectangle, labels=labels)

        assert len(annotation_scene_entity.annotations) == 2

        annotation_scene_entity.append_annotation(annotation)
        assert len(annotation_scene_entity.annotations) == 3

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_annotation_scene_entity_append_annotations(self):
        """
        <b>Description:</b>
        Check Annotation append_annotations method

        <b>Input data:</b>
        Initialized instance of AnnotationSceneEntity

        <b>Expected results:</b>
        Test passes if AnnotationSceneEntity append_annotations method returns correct values

        <b>Steps</b>
        1. Create AnnotationSceneEntity instances
        2. Check returning value of append_annotations method
        """

        annotation_scene_entity = self.annotation_scene_entity

        annotation_scene_entity.append_annotations(self.annotations)
        assert len(annotation_scene_entity.annotations) == 6

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_annotation_scene_entity_get_labels(self):
        """
        <b>Description:</b>
        Check Annotation get_labels method

        <b>Input data:</b>
        Initialized instance of AnnotationSceneEntity

        <b>Expected results:</b>
        Test passes if AnnotationSceneEntity get_labels method returns correct values

        <b>Steps</b>
        1. Create AnnotationSceneEntity instances
        2. Check returning value of get_labels method
        """

        annotation_scene_entity = self.annotation_scene_entity

        assert len(annotation_scene_entity.get_labels()) == 1
        assert "name=tree" in str(annotation_scene_entity.get_labels())

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_annotation_scene_entity_get_label_ids(self):
        """
        <b>Description:</b>
        Check Annotation get_label_ids method

        <b>Input data:</b>
        Initialized instance of AnnotationSceneEntity

        <b>Expected results:</b>
        Test passes if AnnotationSceneEntity get_label_ids method returns correct values

        <b>Steps</b>
        1. Create AnnotationSceneEntity instances
        2. Check returning value of get_label_ids method
        """

        annotation_scene_entity = self.annotation_scene_entity

        assert annotation_scene_entity.get_label_ids() == {ID()}

        bus = LabelEntity(id=ID(123456789), name="bus", domain=Domain.DETECTION)
        bus_label = ScoredLabel(bus)
        labels = [bus_label]
        annotation = Annotation(shape=self.rectangle, labels=labels)
        annotation_scene_entity.append_annotation(annotation)

        assert annotation_scene_entity.get_label_ids() == {ID(), ID(123456789)}


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestNullAnnotationSceneEntity:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_null_annotation_scene_entity(self):
        """
        <b>Description:</b>
        Check that NullAnnotationSceneEntity

        <b>Input data:</b>
        NullAnnotationSceneEntity class

        <b>Expected results:</b>
        Test passes if the NullAnnotationSceneEntity return correct values

        <b>Steps</b>
        1. Create NullAnnotationSceneEntity instances
        2. Check default values
        """

        null_annotation = NullAnnotationSceneEntity()

        assert null_annotation.id_ == ID()
        assert null_annotation.kind == AnnotationSceneKind.NONE
        assert null_annotation.editor_name == ""
        assert type(null_annotation.creation_date) == datetime.datetime
        assert null_annotation.annotations == []
        assert repr(null_annotation) == "NullAnnotationSceneEntity()"
