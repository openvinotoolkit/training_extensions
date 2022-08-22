#
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


import json
from random import randint

import pytest

from otx.api.entities.id import ID
from otx.api.entities.label import Color, Domain, LabelEntity
from otx.api.entities.label_schema import (
    LabelGraph,
    LabelGroup,
    LabelGroupType,
    LabelSchemaEntity,
    LabelTree,
)
from otx.api.serialization.datetime_mapper import DatetimeMapper
from otx.api.serialization.label_mapper import (
    ColorMapper,
    LabelGraphMapper,
    LabelGroupMapper,
    LabelMapper,
    LabelSchemaMapper,
    label_schema_to_bytes,
)
from otx.api.utils.time_utils import now
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestColorMapper:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_color_serialization(self):
        """
        This test serializes Color and checks serialized representation.
        Then it compares deserialized Color with original one.
        """

        red = randint(0, 255)  # nosec
        green = randint(0, 255)  # nosec
        blue = randint(0, 255)  # nosec
        alpha = randint(0, 255)  # nosec
        color = Color(red, green, blue, alpha)
        serialized = ColorMapper.forward(color)
        assert serialized == {"red": red, "green": green, "blue": blue, "alpha": alpha}

        deserialized = ColorMapper.backward(serialized)
        assert color == deserialized


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestLabelEntityMapper:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_entity_serialization(self):
        """
        This test serializes LabelEntity and checks serialized representation.
        Then it compares deserialized LabelEntity with original one.
        """

        cur_date = now()
        red = randint(0, 255)  # nosec
        green = randint(0, 255)  # nosec
        blue = randint(0, 255)  # nosec
        alpha = randint(0, 255)  # nosec

        label = LabelEntity(
            name="my_label",
            domain=Domain.DETECTION,
            color=Color(red, green, blue, alpha),
            hotkey="ctrl+1",
            creation_date=cur_date,
            is_empty=False,
            id=ID("0000213"),
        )
        serialized = LabelMapper.forward(label)

        assert serialized == {
            "_id": "0000213",
            "name": "my_label",
            "color": {"red": red, "green": green, "blue": blue, "alpha": alpha},
            "hotkey": "ctrl+1",
            "domain": "DETECTION",
            "creation_date": DatetimeMapper.forward(cur_date),
            "is_empty": False,
            "is_anomalous": False,
        }

        deserialized = LabelMapper.backward(serialized)
        assert label == deserialized


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestLabelSchemaEntityMapper:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_flat_label_schema_serialization(self):
        """
        This test serializes flat LabelSchema and checks serialized representation.
        Then it compares deserialized LabelSchema with original one.
        """

        cur_date = now()
        names = ["cat", "dog", "mouse"]
        colors = [
            Color(
                randint(0, 255),  # nosec
                randint(0, 255),  # nosec
                randint(0, 255),  # nosec
                randint(0, 255),  # nosec
            )  # nosec  # noqa
            for _ in range(3)
        ]
        labels = [
            LabelEntity(
                name=name,
                domain=Domain.CLASSIFICATION,
                creation_date=cur_date,
                id=ID(i),
                color=colors[i],
            )
            for i, name in enumerate(names)
        ]
        label_schema = LabelSchemaEntity.from_labels(labels)
        serialized = LabelSchemaMapper.forward(label_schema)

        assert serialized == {
            "label_tree": {"type": "tree", "directed": True, "nodes": [], "edges": []},
            "label_groups": [
                {
                    "_id": label_schema.get_groups()[0].id_,
                    "name": "from_label_list",
                    "label_ids": ["0", "1", "2"],
                    "relation_type": "EXCLUSIVE",
                }
            ],
            "all_labels": {
                "0": {
                    "_id": "0",
                    "name": "cat",
                    "color": ColorMapper.forward(colors[0]),
                    "hotkey": "",
                    "domain": "CLASSIFICATION",
                    "creation_date": DatetimeMapper.forward(cur_date),
                    "is_empty": False,
                    "is_anomalous": False,
                },
                "1": {
                    "_id": "1",
                    "name": "dog",
                    "color": ColorMapper.forward(colors[1]),
                    "hotkey": "",
                    "domain": "CLASSIFICATION",
                    "creation_date": DatetimeMapper.forward(cur_date),
                    "is_empty": False,
                    "is_anomalous": False,
                },
                "2": {
                    "_id": "2",
                    "name": "mouse",
                    "color": ColorMapper.forward(colors[2]),
                    "hotkey": "",
                    "domain": "CLASSIFICATION",
                    "creation_date": DatetimeMapper.forward(cur_date),
                    "is_empty": False,
                    "is_anomalous": False,
                },
            },
        }

        deserialized = LabelSchemaMapper.backward(serialized)
        assert label_schema == deserialized

        # Checking value returned by "label_schema_to_bytes" function
        expected_label_schema_to_bytes = json.dumps(serialized, indent=4).encode()
        actual_label_schema_to_bytes = label_schema_to_bytes(label_schema)
        assert actual_label_schema_to_bytes == expected_label_schema_to_bytes


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestLabelGroupMapper:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_group_serialization(self):
        """
        This test serializes flat LabelGroup and checks serialized representation.
        Then it compares deserialized LabelGroup with original one.
        """

        names = ["cat", "dog", "mouse"]
        labels = [
            LabelEntity(
                name=name,
                domain=Domain.CLASSIFICATION,
                id=ID(str(i)),
            )
            for i, name in enumerate(names)
        ]
        label_group = LabelGroup(name="Test LabelGroup", labels=labels, group_type=LabelGroupType.EMPTY_LABEL)
        serialized = LabelGroupMapper.forward(label_group)
        assert serialized == {
            "_id": label_group.id_,
            "name": "Test LabelGroup",
            "label_ids": ["0", "1", "2"],
            "relation_type": "EMPTY_LABEL",
        }
        all_labels = {ID(str(i)): labels[i] for i in range(3)}

        deserialized = LabelGroupMapper.backward(instance=serialized, all_labels=all_labels)
        assert deserialized == label_group


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestLabelGraphMapper:
    label_0 = LabelEntity(name="label_0", domain=Domain.SEGMENTATION, id=ID("0"))
    label_0_1 = LabelEntity(name="label_0_1", domain=Domain.SEGMENTATION, id=ID("0_1"))
    label_0_2 = LabelEntity(name="label_0_2", domain=Domain.SEGMENTATION, id=ID("0_2"))
    label_0_1_1 = LabelEntity(name="label_0_1_1", domain=Domain.SEGMENTATION, id=ID("0_1_1"))
    label_0_1_2 = LabelEntity(name="label_0_1_2", domain=Domain.SEGMENTATION, id=ID("0_1_2"))
    label_0_2_1 = LabelEntity(name="label_0_2_1", domain=Domain.SEGMENTATION, id=ID("0_2_1"))

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_graph_forward(self):
        """
        <b>Description:</b>
        Check "LabelGraphMapper" class "forward" method

        <b>Input data:</b>
        "LabelGraph" and "LabelTree" objects

        <b>Expected results:</b>
        Test passes if dictionary returned by "forward" method is equal to expected

        <b>Steps</b>
        1. Check dictionary returned by "forward" method for "LabelGraph" object
        2. Check dictionary returned by "forward" method for "LabelTree" object
        """
        # Checking dictionary returned by "forward" for "LabelGraph"
        label_graph = LabelGraph(directed=False)
        label_graph.add_edges(
            [
                (self.label_0, self.label_0_1),
                (self.label_0, self.label_0_2),
                (self.label_0_1, self.label_0_1_1),
                (self.label_0_1, self.label_0_1_2),
                (self.label_0_1_1, self.label_0_1_2),
            ]
        )
        forward = LabelGraphMapper.forward(label_graph)
        assert forward == {
            "type": "graph",
            "directed": False,
            "nodes": ["0", "0_1", "0_2", "0_1_1", "0_1_2"],
            "edges": [
                ("0", "0_1"),
                ("0", "0_2"),
                ("0_1", "0_1_1"),
                ("0_1", "0_1_2"),
                ("0_1_1", "0_1_2"),
            ],
        }
        # Checking dictionary returned by "forward" for "LabelTree"
        label_tree = LabelTree()
        for parent, child in [
            (self.label_0, self.label_0_1),
            (self.label_0, self.label_0_2),
            (self.label_0_1, self.label_0_1_1),
            (self.label_0_1, self.label_0_1_2),
            (self.label_0_2, self.label_0_2_1),
        ]:
            label_tree.add_child(parent, child)
        forward = LabelGraphMapper.forward(label_tree)
        assert forward == {
            "type": "tree",
            "directed": True,
            "nodes": ["0_1", "0", "0_2", "0_1_1", "0_1_2", "0_2_1"],
            "edges": [
                ("0_1", "0"),
                ("0_2", "0"),
                ("0_1_1", "0_1"),
                ("0_1_2", "0_1"),
                ("0_2_1", "0_2"),
            ],
        }

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_graph_backward(self):
        """
        <b>Description:</b>
        Check "LabelGraphMapper" class "backward" method

        <b>Input data:</b>
        Dictionary object to deserialize, labels list

        <b>Expected results:</b>
        Test passes if "LabelGraph" or "LabelTree" object returned by "backward" method is equal to expected

        <b>Steps</b>
        1. Check dictionary returned by "backward" method for "LabelGraph" object
        2. Check dictionary returned by "backward" method for "LabelTree" object
        3. Check that "ValueError" exception is raised when unsupported type is specified as "type" key in dictionary
        object of "instance" parameter for "backward" method
        """
        # Checking dictionary returned by "backward" for "LabelGraph"
        forward = {
            "type": "graph",
            "directed": False,
            "nodes": ["0", "0_1", "0_2", "0_1_1"],
            "edges": [("0", "0_1"), ("0", "0_2"), ("0_1", "0_1_1"), ("0_1_1", "0_2")],
        }
        labels = {
            ID("0"): self.label_0,
            ID("0_1"): self.label_0_1,
            ID("0_2"): self.label_0_2,
            ID("0_1_1"): self.label_0_1_1,
        }
        expected_backward = LabelGraph(directed=False)
        expected_backward.add_edges(
            [
                (self.label_0, self.label_0_1),
                (self.label_0, self.label_0_2),
                (self.label_0_1, self.label_0_1_1),
                (self.label_0_1_1, self.label_0_2),
            ]
        )
        actual_backward = LabelGraphMapper.backward(instance=forward, all_labels=labels)
        assert actual_backward == expected_backward
        # Checking dictionary returned by "backward" for "LabelTree"
        forward = {
            "type": "tree",
            "directed": True,
            "nodes": ["0_1", "0", "0_2", "0_1_1", "0_2_1"],
            "edges": [("0_1", "0"), ("0_2", "0"), ("0_1_1", "0_1"), ("0_2_1", "0_2")],
        }
        labels = {
            ID("0"): self.label_0,
            ID("0_1"): self.label_0_1,
            ID("0_2"): self.label_0_2,
            ID("0_1_1"): self.label_0_1_1,
            ID("0_1_2"): self.label_0_1_2,
            ID("0_2_1"): self.label_0_2_1,
        }
        expected_backward = LabelTree()
        for parent, child in [
            (self.label_0, self.label_0_1),
            (self.label_0, self.label_0_2),
            (self.label_0_1, self.label_0_1_1),
            (self.label_0_2, self.label_0_2_1),
        ]:
            expected_backward.add_child(parent, child)
        actual_backward = LabelGraphMapper.backward(instance=forward, all_labels=labels)
        assert actual_backward == expected_backward
        # Checking "ValueError" exception raised when unsupported type specified as "type" in dictionary "instance" for
        # "backward"
        forward = {
            "type": "rectangle",
            "directed": True,
            "nodes": ["0_1", "0", "0_2", "0_1_1", "0_2_1"],
            "edges": [("0_1", "0"), ("0_2", "0"), ("0_1_1", "0_1"), ("0_2_1", "0_2")],
        }
        with pytest.raises(ValueError):
            LabelGraphMapper.backward(instance=forward, all_labels=labels)
