#
# INTEL CONFIDENTIAL
#
# Copyright (C) 2021 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials, and
# your use of them is governed by the express license under which they were provided to
# you ("License"). Unless the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit this software or the related documents
# without Intel's prior written permission.
#
# This software and the related documents are provided as is,
# with no express or implied warranties, other than those that are expressly stated
# in the License.
#


from random import randint

from ote_sdk.entities.id import ID
from ote_sdk.entities.label import Color, Domain, LabelEntity
from ote_sdk.entities.label_schema import LabelSchemaEntity
from ote_sdk.serialization.label_mapper import (
    ColorMapper,
    LabelMapper,
    LabelSchemaMapper,
)
from ote_sdk.utils.time_utils import now


def test_color_serialization():
    """
    This test serializes Color and checks serialized representation.
    Then it compares deserialized Color with original one.
    """

    red = randint(0, 255)
    green = randint(0, 255)
    blue = randint(0, 255)
    alpha = randint(0, 255)
    color = Color(red, green, blue, alpha)
    serialized = ColorMapper.forward(color)
    assert serialized == {"red": red, "green": green, "blue": blue, "alpha": alpha}

    deserialized = ColorMapper.backward(serialized)
    assert color == deserialized


def test_label_entity_serialization():
    """
    This test serializes LabelEntity and checks serialized representation.
    Then it compares deserialized LabelEntity with original one.
    """

    cur_date = now()
    red = randint(0, 255)
    green = randint(0, 255)
    blue = randint(0, 255)
    alpha = randint(0, 255)

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
        "creation_date": cur_date,
        "is_empty": False,
    }

    deserialized = LabelMapper.backward(serialized)
    assert label == deserialized


def test_flat_label_schema_serialization():
    """
    This test serializes flat LabelSchema and checks serialized representation.
    Then it compares deserialized LabelSchema with original one.
    """

    cur_date = now()
    names = ["cat", "dog", "mouse"]
    colors = [
        Color(randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255))
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
    label_shema = LabelSchemaEntity.from_labels(labels)
    serialized = LabelSchemaMapper.forward(label_shema)

    assert serialized == {
        "label_tree": {"type": "tree", "directed": True, "nodes": [], "edges": []},
        "exclusivity_graph": {
            "type": "graph",
            "directed": False,
            "nodes": [],
            "edges": [],
        },
        "label_groups": [
            {
                "_id": label_shema.get_groups()[0].id,
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
                "creation_date": cur_date,
                "is_empty": False,
            },
            "1": {
                "_id": "1",
                "name": "dog",
                "color": ColorMapper.forward(colors[1]),
                "hotkey": "",
                "domain": "CLASSIFICATION",
                "creation_date": cur_date,
                "is_empty": False,
            },
            "2": {
                "_id": "2",
                "name": "mouse",
                "color": ColorMapper.forward(colors[2]),
                "hotkey": "",
                "domain": "CLASSIFICATION",
                "creation_date": cur_date,
                "is_empty": False,
            },
        },
    }

    deserialized = LabelSchemaMapper.backward(serialized)
    assert label_shema == deserialized
