"""This module implements the LabelSchema entity"""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from enum import Enum
from typing import List, Optional, Sequence

from bson import ObjectId

from ote_sdk.entities.graph import Graph, MultiDiGraph
from ote_sdk.entities.id import ID
from ote_sdk.entities.label import LabelEntity
from ote_sdk.entities.scored_label import ScoredLabel


class LabelGroupExistsException(ValueError):
    """Exception thrown if the LabelGroup already exists"""


class LabelGroupDoesNotExistException(ValueError):
    """Exception thrown if the LabelGroup does not exist"""


class LabelGroupType(Enum):
    """Enum to indicate the LabelGroupType"""

    EXCLUSIVE = 1
    EMPTY_LABEL = 2


class LabelGroup:
    """
    A label group which has exclusive (multiclass) or contains the empty label.
    Non-exclusive (multilabel) relationships are represented by multiple (exclusive)
    label groups.

    The labels have to be from one task.

    :param id: ID of the LabelGroup. If no ID is provided, a new ObjectId() will be
               assigned
    :param name: Descriptive name of the label group
    :param labels: Labels that form the group
    :param group_type: EXCLUSIVE or EMPTY_LABEL
    """

    # pylint: disable=redefined-builtin
    def __init__(
        self,
        name: str,
        labels: Sequence[LabelEntity],
        group_type: LabelGroupType = LabelGroupType.EXCLUSIVE,
        id: ID = None,
    ):
        self.id = ID(ObjectId()) if id is None else id

        self.labels = sorted(labels, key=lambda x: x.id)
        self.name = name
        self.group_type = group_type

    @property
    def minimum_label_id(self) -> ID:
        """
        Returns the minimum (oldest) label ID, which is the first label in self.labels
        since this list is sorted
        """
        return self.labels[0].id

    def remove_label(self, label: LabelEntity) -> None:
        """
        remove label from label group if it exists in the group

        :param label: label to remove
        """
        if label in self.labels:
            self.labels.remove(label)

    def is_single_label(self) -> bool:
        """
        Returns True if the label group only contains one label

        :return:
        """
        return len(self.labels) == 1

    def __eq__(self, other: object):
        if not isinstance(other, LabelGroup):
            return False
        return self.id == other.id and (
            set(self.labels) == set(other.labels)
            and self.group_type == other.group_type
        )

    def __repr__(self) -> str:
        return (
            f"LabelGroup(id={self.id}, name={self.name}, group_type={self.group_type},"
            f" labels={self.labels})"
        )


class LabelGraph(Graph):
    """
    Represents connectivity between labels as a graph. For example exclusivity or
    hierarchy.

    :param directed: whether the relationships are directed or undirected (symmetrical)
    """

    def __init__(self, directed: bool):
        super().__init__(directed)

    def add_edges(self, edges):
        """
        Add edges between Labels
        """
        self._graph.add_edges_from(edges)

    @property
    def num_labels(self):
        return self.num_nodes()

    @property
    def type(self):
        """
        Returns the type of the LabelGraph
        """
        return "graph"

    def subgraph(self, labels: Sequence[LabelEntity]) -> "LabelGraph":
        """
        Return the subgraph containing the given labels.
        """
        new_graph = LabelGraph(self.directed)
        new_graph.set_graph(self.get_graph().subgraph(labels).copy())
        return new_graph

    def __eq__(self, other) -> bool:
        if isinstance(other, LabelGraph):
            return super().__eq__(other)
        return False


class LabelTree(MultiDiGraph):
    """
    Represents a hierarchy of labels in the form a tree

    The tree is represented by a directed graph
    """

    def __init__(self) -> None:
        super().__init__()

        self.__topological_order_cache: Optional[List[LabelEntity]] = None

    def add_edge(self, node1, node2, edge_value=None):
        super().add_edge(node1, node2, edge_value)
        self.clear_topological_cache()

    def add_node(self, node):
        super().add_node(node)
        self.clear_topological_cache()

    def add_edges(self, edges):
        """
        Add edges between Labels
        """
        self._graph.add_edges_from(edges)
        self.clear_topological_cache()

    def remove_node(self, node):
        super().remove_node(node)
        self.clear_topological_cache()

    @property
    def num_labels(self):
        return self.num_nodes()

    def clear_topological_cache(self):
        """
        Clear the internal cache of the list of labels sorted in topological order.

        This function should be called if the topology of the graph has changed to
            prevent the cache from being stale.
        Note that it is automatically called when modifying the topology through the
            methods provided by this class.
        """
        self.__topological_order_cache = None

    def get_labels_in_topological_order(self) -> List[LabelEntity]:
        """
        Return a list of the labels in this graph sorted in topological order.

        To avoid performance issues, the output of this function is cached.

        :return: sorted list of labels
        """
        if self.__topological_order_cache is None:
            # TODO: It seems that we are storing the edges the wrong way around.
            #       To work around this issue, we have to reverse the sorted list.
            self.__topological_order_cache = list(
                reversed(list(self.topological_sort()))
            )

        return self.__topological_order_cache

    @property
    def type(self):
        """
        Returns the type of the LabelTree
        """
        return "tree"

    def add_child(self, parent: LabelEntity, child: LabelEntity):
        """
        Add a `child` Label to `parent`
        """
        self.add_edge(child, parent)
        self.clear_topological_cache()

    def get_parent(self, label: LabelEntity) -> Optional[LabelEntity]:
        """
        Returns the parent of `label`.

        :return: the parent if it has one otherwise None
        """
        result = self.neighbors(label)
        return result[0] if len(result) > 0 else None

    def get_children(self, parent: LabelEntity) -> List[LabelEntity]:
        """Returns children of `parent`"""
        if parent not in self._graph.nodes:
            return []
        return list(self._graph.predecessors(parent))  # pylint: disable=no-member

    def get_descendants(self, parent: LabelEntity) -> List[LabelEntity]:
        """Returns descendants (children and children of children, etc.) of `parent`"""
        return self.descendants(parent)

    def get_siblings(self, label: LabelEntity) -> List[LabelEntity]:
        """Returns the siblings of a label"""
        parent = self.get_parent(label)
        if parent is None:
            siblings = []
        else:
            siblings = [
                u
                for u, v in self._graph.in_edges(parent)  # pylint: disable=no-member
                if u != label
            ]
        return siblings

    def get_ancestors(self, label: LabelEntity) -> List[LabelEntity]:
        """
        Returns ancestors of `label`, including self
        """
        result = []
        parent: Optional[LabelEntity] = label
        while parent is not None:
            result.append(parent)
            parent = self.get_parent(parent)
        return result

    def subgraph(self, labels: Sequence[LabelEntity]) -> "LabelTree":
        """
        Return the subgraph containing the given labels.
        """
        new_graph = LabelTree()
        new_graph.set_graph(self.get_graph().subgraph(labels).copy())
        return new_graph

    def __eq__(self, other) -> bool:
        if isinstance(other, LabelTree):
            return super().__eq__(other)
        return False


class LabelSchemaEntity:
    """
    This class represents the relationships of labels.

    This class currently keeps track of the following relationships:

    - parent/child label relationship
    - label group relationships

    :param label_tree: a hierarchy of labels represented as a tree
    :param label_groups: list of groups of labels that form logical groups. E.g. a group
        of mutually exclusive labels.
    """

    # pylint: disable=too-many-public-methods, too-many-arguments
    def __init__(
        self,
        label_tree: LabelTree = None,
        label_groups: List[LabelGroup] = None,
    ):
        if label_tree is None:
            label_tree = LabelTree()
        self.label_tree = label_tree

        if label_groups is None:
            label_groups = []
        self._groups = label_groups

    def get_labels(self, include_empty) -> List[LabelEntity]:
        """
        Get the labels in the label schema

        :param include_empty: flag determining whether to include empty labels
        :return: list of all labels in the label schema
        """
        labels = {
            label
            for group in self._groups
            for label in group.labels
            if include_empty or not label.is_empty
        }
        return sorted(list(labels), key=lambda x: x.id)

    def get_groups(self, include_empty: bool = False) -> List[LabelGroup]:
        """
        Get the label groups in the label schema

        :param include_empty: flag determining whether to include empty label groups
        :return: list of all label groups in the label schema
        """
        if include_empty:
            return self._groups

        return [
            group
            for group in self._groups
            if group.group_type != LabelGroupType.EMPTY_LABEL
        ]

    def add_group(self, label_group: LabelGroup):
        """
        Adding a group to label schema.

        :param label_group: label group to add
        :return:
        """
        if label_group.name in [group.name for group in self._groups]:
            raise LabelGroupExistsException(
                f"group with '{label_group.name}' exists, "
                f"use add_to_group_by_group_name instead"
            )
        self.__append_group(label_group)

    def add_child(self, parent: LabelEntity, child: LabelEntity):
        """
        Add a `child` Label to `parent`
        """
        parent = self.__get_label(parent)
        child = self.__get_label(child)
        self.label_tree.add_child(parent, child)

    def get_parent(self, label: LabelEntity) -> Optional[LabelEntity]:
        """
        Returns the parent of `label`.

        :return: the parent if it has one otherwise None
        """
        label = self.__get_label(label)
        return self.label_tree.get_parent(label)

    def get_label_ids(self, include_empty) -> List[ID]:
        """
        Returns a list of label ids that are in the LabelSchema
        :param include_empty: Include empty label id or not
        """
        label_ids = {
            label.id
            for group in self._groups
            for label in group.labels
            if include_empty or not label.is_empty
        }
        return sorted(list(label_ids))

    def get_label_group_by_name(self, group_name: str) -> Optional[LabelGroup]:
        """
        Get the label group by the passed group_name
        :param group_name:
        :return:
        """
        for label_group in self._groups:
            if group_name == label_group.name:
                return label_group
        return None

    def get_exclusive_groups(self) -> List[LabelGroup]:
        """
        Returns exclusive groups in the LabelSchema
        """

        return [
            group
            for group in self._groups
            if group.group_type == LabelGroupType.EXCLUSIVE
        ]

    def add_labels_to_group_by_group_name(
        self, group_name: str, labels: Sequence[LabelEntity]
    ):
        """
        Adds `labels` to group named `group_name`

        :param labels: list of Label
        :param group_name: group name

        :raises LabelGroupDoesNotExistException: This is raised if the group does not
            exist
        """
        group = self.get_label_group_by_name(group_name)
        if group is not None:
            group.labels.extend(labels)
        else:
            raise LabelGroupDoesNotExistException(
                f"group with name '{group_name}' does not exist, cannot add"
            )

    def __append_group(self, label_group: LabelGroup):
        """
        Convenience function for appending `label_group` to the necessary internal data
            structures
        :param label_group:
        """
        if label_group not in self._groups:
            self._groups.append(label_group)

    def are_exclusive(self, label1: LabelEntity, label2: LabelEntity) -> bool:
        """
        Returns whether `label` and `label2` are mutually exclusive.
        or if
        """
        return label2 in self.get_labels_exclusive_to(label1)

    def get_children(self, parent: LabelEntity) -> List[LabelEntity]:
        """Return a list of the children of the passed parent Label"""
        parent = self.__get_label(parent)
        return self.label_tree.get_children(parent)

    def get_siblings_in_group(self, label: LabelEntity) -> List[LabelEntity]:
        """Return a list of the 'siblings', which are all labels within the same group as a label"""
        containing_group = self.get_group_containing_label(label)
        if containing_group is None:
            return []
        return [
            label_iter
            for label_iter in containing_group.labels
            if not label_iter == label
        ]

    def get_descendants(self, parent: LabelEntity) -> List[LabelEntity]:
        """Returns descendants (children and children of children, etc.) of `parent`"""
        parent = self.__get_label(parent)
        return self.label_tree.get_descendants(parent)

    def get_ancestors(self, label: LabelEntity) -> List[LabelEntity]:
        """
        Returns ancestors of `label`, including self
        """
        label = self.__get_label(label)
        return self.label_tree.get_ancestors(label)

    def get_group_containing_label(self, label: LabelEntity) -> Optional[LabelGroup]:
        """
        Returns the label group which contains the label.

        :param label: the query label
        :return:
        """
        label = self.__get_label(label)
        for group in self._groups:
            if label in group.labels:
                return group
        return None

    def get_labels_exclusive_to(self, label: LabelEntity) -> List[LabelEntity]:
        """Returns a list of labels that are exclusive to the passed label."""
        return self.__get_exclusivity_recursion(label, True)

    def __get_exclusivity_recursion(
        self, label, add_children: bool = False
    ) -> List[LabelEntity]:
        """
        Recursively computes all labels exclusive to a label. A label is exclusive with:
        - All labels in the same group
        - All children of labels in the same group
        - All labels in the same group as any of the label's ancestors
        - All children of labels in the same group as any of the label's ancestors ancestors

        :param label: The label to get exclusive labels for
        :return: List of labels exclusive to the label.
        """
        output = []

        # Add all labels in the same group
        siblings = self.get_siblings_in_group(label)
        output += siblings

        # Add all children of labels in the same group
        if add_children:
            for sibling in siblings:
                output += self.get_children(sibling)

        # Do the same for the parent of the label
        if self.get_parent(label) is not None:
            output += self.__get_exclusivity_recursion(self.get_parent(label))
        return output

    @staticmethod
    def __get_label(label) -> LabelEntity:
        """
        Returns Label object from possibly non-label object

        :param label:

        :return:
        """
        if isinstance(label, ScoredLabel):
            returned_label = label.get_label()
        elif isinstance(label, LabelEntity):
            returned_label = label
        else:
            raise ValueError("Input of __get_label is not of type Label or ScoredLabel")
        return returned_label  # type: ignore

    def __repr__(self) -> str:
        return f"LabelSchemaEntity(label_groups={self._groups})"

    def __eq__(self, other) -> bool:
        if isinstance(other, LabelSchemaEntity):
            return self.label_tree == other.label_tree and self.get_groups(
                include_empty=True
            ) == other.get_groups(include_empty=True)
        return False

    @classmethod
    def from_labels(cls, labels: Sequence[LabelEntity]):
        """
        Create LabelSchemaEntity from a list of exclusive labels

        :param labels: list of labels
        :return: LabelSchemaEntity from the given labels
        """
        label_group = LabelGroup(name="from_label_list", labels=labels)
        return LabelSchemaEntity(label_groups=[label_group])
