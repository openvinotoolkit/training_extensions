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

""" This module contains the mapper for label related entities """

from typing import Union

from ote_sdk.entities.color import Color
from ote_sdk.entities.label import Domain, LabelEntity
from ote_sdk.entities.label_schema import (
    LabelGraph,
    LabelGroup,
    LabelGroupType,
    LabelSchemaEntity,
    LabelTree,
)

from .id_mapper import IDMapper
from .primitive_mapper import DatetimeMapper


class ColorMapper:
    """
    This class maps a `Color` entity to a serialized dictionary, and vice versa
    """

    def forward(self, instance: Color) -> dict:
        return {
            "red": instance.red,
            "green": instance.green,
            "blue": instance.blue,
            "alpha": instance.alpha,
        }

    def backward(self, instance: dict) -> Color:
        return Color(
            instance["red"], instance["green"], instance["blue"], instance["alpha"]
        )


class LabelMapper:
    """
    This class maps a `Label` entity to a serialized dictionary, and vice versa
    """

    def __init__(self) -> None:
        pass

    def forward(
        self,
        instance: LabelEntity,
    ) -> dict:
        return {
            "_id": IDMapper().forward(instance.id),
            "name": instance.name,
            "color": ColorMapper().forward(instance.color),
            "hotkey": instance.hotkey,
            "domain": str(instance.domain),
            "creation_date": DatetimeMapper().forward(instance.creation_date),
            "is_empty": instance.is_empty,
        }

    def backward(self, instance: dict) -> LabelEntity:
        label_id = IDMapper().backward(instance["_id"])

        domain = instance.get("domain")
        label_domain = Domain[domain]

        label = LabelEntity(
            id=label_id,
            name=instance["name"],
            color=ColorMapper().backward(instance["color"]),
            hotkey=instance.get("hotkey", ""),
            domain=label_domain,
            creation_date=DatetimeMapper().backward(
                instance.get("creation_date", None)
            ),
            is_empty=instance.get("is_empty", False),
        )
        return label


class LabelGroupMapper:
    """
    This class maps a `LabelGroup` entity to a serialized dictionary, and vice versa
    """

    def forward(self, instance: LabelGroup) -> dict:
        return {
            "_id": IDMapper().forward(instance.id),
            "name": instance.name,
            "label_ids": [IDMapper().forward(label.id) for label in instance.labels],
            "relation_type": instance.group_type.name,
        }

    def backward(self, instance: dict, all_labels: dict) -> LabelGroup:
        return LabelGroup(
            id=IDMapper().backward(instance["_id"]),
            name=instance["name"],
            group_type=LabelGroupType[instance["relation_type"]],
            labels=[
                all_labels.get(IDMapper().backward(label_id))
                for label_id in instance["label_ids"]
            ],
        )


class LabelGraphMapper:
    """
    This class maps a `LabelGraph` entity to a serialized dictionary, and vice versa
    """

    def forward(self, instance: Union[LabelGraph, LabelTree]) -> dict:
        return {
            "type": instance.type,
            "directed": instance.directed,
            "nodes": [IDMapper().forward(label.id) for label in instance.nodes],
            "edges": [
                (IDMapper().forward(edge[0].id), IDMapper().forward(edge[1].id))
                for edge in instance.edges
            ],
        }

    def backward(
        self, instance: dict, all_labels: dict
    ) -> Union[LabelTree, LabelGraph]:

        instance_type = instance["type"]
        if instance_type == "tree":
            output = LabelTree()
        elif instance_type == "graph":
            output = LabelGraph(instance["directed"])
        else:
            raise ValueError(f"Unsupported type `{instance_type}` for label graph")

        label_map = {
            label_id: all_labels.get(IDMapper().backward(label_id))
            for label_id in instance["nodes"]
        }
        for label in label_map.values():
            output.add_node(label)
        for edge in instance["edges"]:
            output.add_edge(label_map[edge[0]], label_map[edge[1]])

        return output


class LabelSchemaMapper:
    """
    This class maps a `LabelSchema` entity to a serialized dictionary, and vice versa
    """

    def forward(
        self,
        instance: LabelSchemaEntity,
    ) -> dict:

        label_groups = [
            LabelGroupMapper().forward(group)
            for group in instance.get_groups(include_empty=True)
        ]

        output_dict = {
            "label_tree": LabelGraphMapper().forward(instance.label_tree),
            "exclusivity_graph": LabelGraphMapper().forward(instance.exclusivity_graph),
            "label_groups": label_groups,
        }

        output_dict["all_labels"] = {
            IDMapper().forward(label.id): LabelMapper().forward(label)
            for label in instance.get_labels(True)
        }

        return output_dict

    def backward(self, instance: dict) -> LabelSchemaEntity:

        all_labels = {
            IDMapper().backward(id): LabelMapper().backward(label)
            for id, label in instance["all_labels"].items()
        }

        exclusivity_graph = LabelGraphMapper().backward(
            instance["exclusivity_graph"], all_labels
        )
        label_tree = LabelGraphMapper().backward(instance["label_tree"], all_labels)
        label_groups = [
            LabelGroupMapper().backward(label_group, all_labels)
            for label_group in instance["label_groups"]
        ]
        output = LabelSchemaEntity(
            exclusivity_graph=exclusivity_graph,
            label_tree=label_tree,
            label_groups=label_groups,
        )
        return output
