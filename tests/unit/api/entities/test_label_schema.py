"""This module tests classes related to LabelSchema"""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
from networkx.classes.reportviews import EdgeDataView, NodeView, OutMultiEdgeDataView

from otx.api.entities.color import Color
from otx.api.entities.id import ID
from otx.api.entities.label import Domain
from otx.api.entities.label_schema import (
    LabelEntity,
    LabelGraph,
    LabelGroup,
    LabelGroupDoesNotExistException,
    LabelGroupExistsException,
    LabelGroupType,
    LabelSchemaEntity,
    LabelTree,
    ScoredLabel,
)
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestLabelSchema:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_tree(self, label_schema_example):
        """
        <b>Description:</b>
        Check that childs and parents of Labels can be found correctly

        <b>Input data:</b>
        A simple LabelTree

        <b>Expected results:</b>
        Test passes if the correct parent and child labels can be found

        <b>Steps</b>
        1. Create LabelTree
        2. Find parents of Labels
        3. Find children of Label
        """
        threat = label_schema_example.new_label_by_name("threat")
        benign = label_schema_example.new_label_by_name("benign")
        gun = label_schema_example.new_label_by_name("gun")
        rifle = label_schema_example.new_label_by_name("rifle")
        bottle = label_schema_example.new_label_by_name("bottle")

        label_tree = LabelTree()
        label_tree.add_child(threat, gun)
        label_tree.add_child(threat, rifle)
        label_tree.add_child(benign, bottle)

        assert label_tree.get_parent(gun) == threat
        assert label_tree.get_parent(rifle) == threat
        assert label_tree.get_parent(bottle) == benign
        assert label_tree.get_parent(threat) is None

        threat_children = label_tree.get_children(threat)
        assert rifle in threat_children

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_exclusive(self, label_schema_example):
        """
        <b>Description:</b>
        Tests that the are_exclusive method properly computes the labels that are exclusive with each other

        <b>Input data:</b>
        Exclusive group: flowering,vegetative,no_plant
        Exclusive group: bee

        <b>Expected results:</b>
        Test passes if the exclusive groups are correctly saved and retrieved

        <b>Steps</b>
        1. Create LabelRelation with exclusive group
        2. Check that relations are exclusive
        3. Check that number of relations is correct
        4. Add new exclusive group
        5. Check that exclusive relations still resolve correctly
        """
        flowering = label_schema_example.flowering
        no_plant = label_schema_example.no_plant
        vegetative = label_schema_example.vegetative

        label_schema = LabelSchemaEntity()
        label_schema.add_group(
            LabelGroup(
                "plant_state",
                [flowering, no_plant, vegetative],
            )
        )

        assert label_schema.are_exclusive(flowering, no_plant)
        assert label_schema.are_exclusive(vegetative, no_plant)
        assert label_schema.are_exclusive(vegetative, flowering)

        bee = label_schema_example.new_label_by_name("bee")
        label_schema.add_group(LabelGroup("bee_state", [bee], LabelGroupType.EXCLUSIVE))

        assert not label_schema.are_exclusive(flowering, bee)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_complex(self, label_schema_example):
        """
        <b>Description:</b>
        Tests that exclusivity is computed correctly in a schema with complex hierarchy

        <b>Input data:</b>
        Exclusive groups: plant, animal
        Exclusive groups: tree, bush (children of plant), mammal, insect (children of animal)
        Empty label

        <b>Expected results:</b>
        Test passes if the are_exclusive method properly computes the exclusivity between labels with various relations

        <b>Steps</b>
        1. Create LabelSchemaEntity with hierarchy
        2. Check that labels are not exclusive with their parent
        3. Check that labels are exclusive with:
         - labels in the same group
         - children of labels in the same group
         - labels in the same group as their parent
         - children of labels in the same group as their parent
        4. Add an empty label and check that it's exclusive with all other labels
        5. Make the empty label the child of a different label, and check that the empty label is not exclusive with
        its parent.
        6. Check that LabelSchema.are_exclusive is symmetric for all cases

        """
        label_schema = LabelSchemaEntity()

        plant = label_schema_example.new_label_by_name("plant")
        animal = label_schema_example.new_label_by_name("animal")
        label_schema.add_group(LabelGroup("organism_type", [plant, animal], LabelGroupType.EXCLUSIVE))

        tree = label_schema_example.new_label_by_name("tree")
        bush = label_schema_example.new_label_by_name("bush")
        label_schema.add_group(LabelGroup("plant_type", [tree, bush], LabelGroupType.EXCLUSIVE))

        label_schema.add_child(parent=plant, child=tree)
        label_schema.add_child(parent=plant, child=bush)

        insect = label_schema_example.new_label_by_name("insect")
        mammal = label_schema_example.new_label_by_name("mammal")
        label_schema.add_group(LabelGroup("animal_type", [insect, mammal], LabelGroupType.EXCLUSIVE))
        label_schema.add_child(parent=animal, child=insect)
        label_schema.add_child(parent=animal, child=mammal)

        assert not label_schema.are_exclusive(plant, tree)
        assert label_schema.are_exclusive(tree, bush)
        assert label_schema.are_exclusive(plant, insect)
        assert label_schema.are_exclusive(tree, animal)
        assert label_schema.are_exclusive(tree, insect)

        # Check that the empty label is exclusive with all labels
        empty_label = label_schema_example.new_label_by_name("empty_label", is_empty=True)
        label_schema.add_group(LabelGroup("empty_label", [empty_label], LabelGroupType.EMPTY_LABEL))
        for label_iter in label_schema.get_labels(include_empty=False):
            assert label_schema.are_exclusive(empty_label, label_iter)

        # Check that the empty label is not exclusive with its parent
        label_schema.add_child(parent=tree, child=empty_label)
        assert not label_schema.are_exclusive(tree, empty_label)

        # Check that label_schema.are_exclusive is symmetric for all cases
        for label_1 in label_schema.get_labels(include_empty=True):
            for label_2 in label_schema.get_labels(include_empty=True):
                assert label_schema.are_exclusive(label_1, label_2) == label_schema.are_exclusive(label_2, label_1)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_add_label_to_group(self, label_schema_example):
        """
        <b>Description:</b>
        Check that new labels can be added to groups

        <b>Input data:</b>
        Empty Label Tree

        <b>Expected results:</b>
        Test passes if an exclusive group can be added to the LabelTree
        as long as a group with the same name does not exist and new labels can be added to a group.

        <b>Steps</b>
        1. Create LabelRelation
        2. Create new label and add to exclusive group
        3. Create new label and attemp to create new group with the same name
        4. Add label to group by group name
        """
        label_schema = LabelSchemaEntity()
        bee = label_schema_example.new_label_by_name("bee")  # indicates presence/absence of bee
        bee_state = LabelGroup("bee_state", [bee], LabelGroupType.EXCLUSIVE)
        label_schema.add_group(bee_state)

        # Try to add an extra bee property as a new exclusive label group, but with the same name
        # as an already existing label group
        flying = label_schema_example.new_label_by_name("flying")
        with pytest.raises(ValueError):
            label_schema.add_group(LabelGroup("bee_state", [flying], LabelGroupType.EXCLUSIVE))

        label_schema.add_labels_to_group_by_group_name("bee_state", [flying])

        assert "bee_state" == bee_state.name
        assert 2 == len(bee_state.labels)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_exclusivity_with_group_addition(self, label_schema_example):
        """
        <b>Description:</b>
        Check that exclusive groups work correctly

        <b>Input data:</b>
        Exclusive Group "Vegetation"
        LabelSchemaExample instance

        <b>Expected results:</b>
        Test passes if a single label and multiple labels can be added to the exclusive group
        and the correct labels can be retrieved

        <b>Steps</b>
        1. Create LabelSchemaEntity with "Vegetation" exclusive group
        2. Add "no_plant" to exclusive group and check that "flowering" is exclusive to "no_plant"
        3. Add "vegetative" to exclusive group and check that "vegetative" is exclusive to "no_plant"
        4. Add "flowering" to exclusive group and check that "no_plant" and "vegetative" are exclusive to "flowering"
        5. Create new LabelRelation instace
        6. Add "no_plant" and "vegetative" to exclusive group "flowering"
        7. Check that "flowering" and "vegetative" are exclusive to "no_plants"
        8. Check that no_plant" and "vegetative" are exclusive to "flowering
        """
        label_schema = LabelSchemaEntity()
        label_group_name = "Vegetation"
        label_schema.add_group(
            LabelGroup(
                label_group_name,
                [label_schema_example.flowering],
                LabelGroupType.EXCLUSIVE,
            )
        )

        label_schema.add_labels_to_group_by_group_name(label_group_name, [label_schema_example.no_plant])
        exclusive_to_no_plants = label_schema.get_labels_exclusive_to(label_schema_example.no_plant)
        assert label_schema_example.flowering in exclusive_to_no_plants

        label_schema.add_labels_to_group_by_group_name(label_group_name, [label_schema_example.vegetative])
        exclusive_to_no_plants = label_schema.get_labels_exclusive_to(label_schema_example.no_plant)
        assert label_schema_example.vegetative in exclusive_to_no_plants

        exclusive_to_flowering = label_schema.get_labels_exclusive_to(label_schema_example.flowering)
        assert label_schema_example.no_plant in exclusive_to_flowering
        assert label_schema_example.vegetative in exclusive_to_flowering

        # new label schema (test adding multiple labels at once)
        label_schema = LabelSchemaEntity()
        label_group_name = "Vegetation"
        label_schema.add_group(
            LabelGroup(
                label_group_name,
                [label_schema_example.flowering],
                LabelGroupType.EXCLUSIVE,
            )
        )
        label_schema.add_labels_to_group_by_group_name(
            label_group_name,
            [label_schema_example.no_plant, label_schema_example.vegetative],
        )

        exclusive_to_no_plants = label_schema.get_labels_exclusive_to(label_schema_example.no_plant)
        assert label_schema_example.flowering in exclusive_to_no_plants
        assert label_schema_example.vegetative in exclusive_to_no_plants

        exclusive_to_flowering = label_schema.get_labels_exclusive_to(label_schema_example.flowering)
        assert label_schema_example.no_plant in exclusive_to_flowering
        assert label_schema_example.vegetative in exclusive_to_flowering

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_labelschema_equality(self, label_schema_example):
        """
        <b>Description:</b>
        Check that LabelSchemaEntity equality works correctly

        <b>Input data:</b>
        LabelSchemaEntity  instances

        <b>Expected results:</b>
        == and != operations work correctly for various inputs

        <b>Steps</b>
        1. Test LabelSchemaEntity equality
        """
        label_schema = LabelSchemaEntity()

        label_group_name = "Vegetation"
        label_schema.add_group(
            LabelGroup(
                label_group_name,
                [label_schema_example.flowering],
                LabelGroupType.EXCLUSIVE,
            )
        )

        label_schema.add_labels_to_group_by_group_name(label_group_name, [label_schema_example.no_plant])

        copy_schema = label_schema
        assert label_schema == copy_schema

        new_schema = LabelSchemaEntity()

        label_group_name = "Vegetation"
        new_schema.add_group(
            LabelGroup(
                label_group_name,
                [label_schema_example.flowering],
                LabelGroupType.EXCLUSIVE,
            )
        )

        new_schema.add_labels_to_group_by_group_name(label_group_name, [label_schema_example.vegetative])

        assert new_schema != label_schema


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestLabelGroupType:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_group_type(self):
        """
        <b>Description:</b>
        Check LabelGroupType Enum class elements

        <b>Expected results:</b>
        Test passes if LabelGroupType Enum class length is equal to expected value and its elements have expected
        sequence numbers
        """
        assert len(LabelGroupType) == 2
        assert LabelGroupType.EXCLUSIVE.value == 1
        assert LabelGroupType.EMPTY_LABEL.value == 2


class Labels:
    def __init__(self):
        self.label_0 = LabelEntity(
            name="Label 0",
            domain=Domain.CLASSIFICATION,
            id=ID("0"),
            color=Color(25, 200, 166),
        )
        self.label_0_1 = LabelEntity(
            name="Label 0_1",
            domain=Domain.DETECTION,
            id=ID("0_1"),
            color=Color(40, 100, 17),
        )
        self.label_0_2 = LabelEntity(
            name="Label 0_2",
            domain=Domain.SEGMENTATION,
            id=ID("0_2"),
            color=Color(30, 80, 40),
        )
        self.label_0_1_3 = LabelEntity(
            name="Label_0_1_3",
            domain=Domain.SEGMENTATION,
            id=ID("0_1_3"),
            color=Color(40, 100, 17),
        )
        self.label_0_2_4 = LabelEntity(
            name="Label_0_2_4",
            domain=Domain.SEGMENTATION,
            id=ID("0_2_4"),
            color=Color(30, 80, 40),
        )
        self.label_0_2_5 = LabelEntity(
            name="Label_0_2_5",
            domain=Domain.SEGMENTATION,
            id=ID("0_2_5"),
            color=Color(30, 80, 40),
        )
        self.no_id_label = LabelEntity(name="No ID Label", domain=Domain.SEGMENTATION)
        self.non_included_label = LabelEntity(name="Label non included to group", domain=Domain.SEGMENTATION)


labels = Labels()


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestLabelGroup:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_group_initialization(self):
        """
        <b>Description:</b>
        Check LabelGroup class object initialization

        <b>Input data:</b>
        LabelGroup object with specified name, labels, group type and ID parameters

        <b>Expected results:</b>
        Test passes if LabelGroup object id, labels, name and group_type attributes return expected values

        <b>Steps</b>
        1. Check id, labels, name and group_type attributes of LabelGroup object with not specified group_type
        parameter
        2. Check id, labels, name and group_type attributes of LabelGroup object with not specified id parameter
        """
        # Checking attributes of LabelGroup object with specified id and not specified group_type parameters
        no_group_type_label_group = LabelGroup(
            name="Type non-specified specified label group",
            labels=[labels.label_0_1, labels.no_id_label, labels.label_0],
            id=ID("1"),
        )
        assert no_group_type_label_group.id_ == "1"
        # Expected ascending sorting of labels
        assert no_group_type_label_group.labels == [
            labels.no_id_label,
            labels.label_0,
            labels.label_0_1,
        ]
        assert no_group_type_label_group.name == "Type non-specified specified label group"
        assert no_group_type_label_group.group_type == LabelGroupType.EXCLUSIVE
        assert isinstance(no_group_type_label_group.minimum_label_id, ID)
        assert no_group_type_label_group.minimum_label_id == ""
        # Checking attributes of LabelGroup object with specified group_type and not specified id parameters
        no_id_label_group = LabelGroup(
            name="ID non-specified Label Group",
            labels=[labels.label_0_1, labels.label_0],
            group_type=LabelGroupType.EMPTY_LABEL,
        )
        # Expected randomly generated ID object with 24 characters as "id" attribute
        assert isinstance(no_id_label_group.id_, ID)
        assert len(no_id_label_group.id_) == 24
        # Expected ascending sorting of labels
        assert no_id_label_group.labels == [labels.label_0, labels.label_0_1]
        assert no_id_label_group.name == "ID non-specified Label Group"
        assert no_id_label_group.group_type == LabelGroupType.EMPTY_LABEL
        assert isinstance(no_id_label_group.minimum_label_id, ID)
        assert no_id_label_group.minimum_label_id == "0"

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_group_remove_label(self):
        """
        <b>Description:</b>
        Check remove_label method of LabelGroup class object

        <b>Input data:</b>
        LabelGroup objects with specified name, labels, group type and id_parameters parameters

        <b>Expected results:</b>
        Test passes if after using remove_label method values of "labels" property, "minimum_label_id" and
        "is_single_label" methods are equal to expected
        """
        label_group = LabelGroup(name="Test Label Group", labels=[labels.label_0, labels.label_0_1])
        assert not label_group.is_single_label()
        # Removing first label in "labels" property and checking values of "labels", "minimum_label_id" and
        # "is_single_label"
        label_group.remove_label(labels.label_0)
        assert label_group.labels == [labels.label_0_1]
        assert label_group.minimum_label_id == "0_1"
        assert label_group.is_single_label()
        # Removing label that not included to LabelGroup object and repeat checks
        label_group.remove_label(labels.non_included_label)
        assert label_group.labels == [labels.label_0_1]
        assert label_group.minimum_label_id == "0_1"
        assert label_group.is_single_label()

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_group_magic_methods(self):
        """
        <b>Description:</b>
        Check __eq__ and __repr__ methods of LabelGroup class object

        <b>Input data:</b>
        LabelGroup class object with specified name, labels, group type and ID parameters

        <b>Expected results:</b>
        Test passes if __eq__ and __repr__ methods return expected values

        <b>Steps</b>
        1. Check __eq__ method for LabelGroup class objects
        2. Check __repr__ method for LabelGroup class objects
        """
        group_labels = [labels.label_0, labels.label_0_1]
        name = "Test Label Group"
        group_id = ID("1")
        label_group = LabelGroup(name=name, labels=group_labels, id=group_id)
        equal_label_group = LabelGroup(name=name, labels=group_labels, id=group_id)
        no_id_specified_label_group = LabelGroup(name=name, labels=group_labels)
        # Checking __eq__ method for equal LabelGroup objects
        assert label_group == equal_label_group
        # Checking equality of LabelGroups with different "name" attributes
        assert label_group == LabelGroup(name="Different name LabelGroup", labels=group_labels, id=group_id)
        # Checking inequality of LabelGroups with different "id" attributes
        assert not label_group == no_id_specified_label_group
        # Checking inequality of LabelGroups with different "labels" attributes
        assert not label_group == LabelGroup(name=name, labels=[labels.label_0], id=group_id)
        # Checking inequality of LabelGroups with different "group_type" attributes
        assert not label_group == LabelGroup(
            name=name,
            labels=group_labels,
            id=group_id,
            group_type=LabelGroupType.EMPTY_LABEL,
        )
        # Checking inequality of LabelGroups with different object of different type
        assert not label_group == str
        # Checking __repr__ method for LabelGroup object with specified id
        assert repr(label_group) == (
            "LabelGroup(id=1, name=Test Label Group, group_type=LabelGroupType.EXCLUSIVE, " f"labels={group_labels})"
        )
        # Checking __repr__ method for LabelGroup object with [] labels and not specified id
        no_labels_no_id_label_group = LabelGroup(
            name="Null labels, random id group",
            labels=[],
            group_type=LabelGroupType.EMPTY_LABEL,
        )
        generated_id = no_labels_no_id_label_group.id_
        assert repr(no_labels_no_id_label_group) == (
            f"LabelGroup(id={generated_id}, name=Null labels, random id group,"
            f" group_type=LabelGroupType.EMPTY_LABEL, labels=[])"
        )


class CommonGraphMethods:
    @staticmethod
    def check_graph_non_list_attributes(expected_attributes_dicts: list) -> None:
        for expected_attribute_dict in expected_attributes_dicts:
            assert expected_attribute_dict.get("attribute") == expected_attribute_dict.get("expected_value")

    @staticmethod
    def check_graph_list_attributes(actual_expected_attributes_dicts: list) -> None:
        for expected_attribute_dict in actual_expected_attributes_dicts:
            attribute = expected_attribute_dict.get("attribute")
            assert isinstance(attribute, expected_attribute_dict.get("expected_type"))
            assert list(attribute) == expected_attribute_dict.get("expected_value")


class Edges:
    def __init__(self):
        self.edge_0_to_0_1 = (labels.label_0, labels.label_0_1)
        self.edge_0_to_0_2 = (labels.label_0, labels.label_0_2)
        self.edge_0_1_to_0_2 = (labels.label_0_1, labels.label_0_2)
        self.edge_0_2_to_0 = (labels.label_0_2, labels.label_0)
        self.edge_0_1_to_0_1_3 = (labels.label_0_1, labels.label_0_1_3)
        self.edge_0_2_to_0_2_4 = (labels.label_0_2, labels.label_0_2_4)
        self.edge_0_2_to_0_2_5 = (labels.label_0_2, labels.label_0_2_5)
        self.edge_0_2_4_to_0_2_5 = (labels.label_0_2_4, labels.label_0_2_5)


edges = Edges()


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestLabelGraph:
    @staticmethod
    def non_directed_label_graph() -> LabelGraph:
        non_directed_graph = LabelGraph(directed=False)
        non_directed_graph.add_edges([edges.edge_0_to_0_1, edges.edge_0_1_to_0_2])
        return non_directed_graph

    @staticmethod
    def directed_label_graph() -> LabelGraph:
        directed_graph = LabelGraph(directed=True)
        directed_graph.add_edges([edges.edge_0_to_0_1, edges.edge_0_1_to_0_2, edges.edge_0_2_to_0])
        return directed_graph

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_graph_initialization(self):
        """
        <b>Description:</b>
        Check LabelGraph class object initialization

        <b>Input data:</b>
        LabelGraph objects with specified "directed" parameter and added edges

        <b>Expected results:</b>
        Test passes if LabelGraph object "directed" attribute and "num_labels" and "type" properties are equal to
        expected

        <b>Steps</b>
        1. Check "directed" attribute and "num_labels" and "type" properties of non-directed LabelGraph object
        2. Check "directed" attribute and "num_labels" and "type" properties of directed LabelGraph object
        """
        # Checks for non-directed LabelGraph
        non_directed_graph = LabelGraph(directed=False)
        # Check for initiated non-directed LabelGraph
        CommonGraphMethods().check_graph_non_list_attributes(
            [
                {"attribute": non_directed_graph.directed, "expected_value": False},
                {"attribute": non_directed_graph.num_labels, "expected_value": 0},
                {"attribute": non_directed_graph.num_nodes(), "expected_value": 0},
                {"attribute": non_directed_graph.type, "expected_value": "graph"},
            ]
        )
        # Check for non-directed LabelGraph with added edges and nodes
        non_directed_graph.add_edges([(labels.label_0, labels.label_0_1), (labels.label_0_1, labels.label_0_2)])
        CommonGraphMethods().check_graph_non_list_attributes(
            [
                {"attribute": non_directed_graph.directed, "expected_value": False},
                {"attribute": non_directed_graph.num_labels, "expected_value": 3},
                {"attribute": non_directed_graph.num_nodes(), "expected_value": 3},
                {"attribute": non_directed_graph.type, "expected_value": "graph"},
            ]
        )
        expected_nodes = [labels.label_0, labels.label_0_1, labels.label_0_2]
        CommonGraphMethods().check_graph_list_attributes(
            [
                {
                    "attribute": non_directed_graph.edges,
                    "expected_type": EdgeDataView,
                    "expected_value": [
                        (labels.label_0, labels.label_0_1, {}),
                        (labels.label_0_1, labels.label_0_2, {}),
                    ],
                },
                {
                    "attribute": non_directed_graph.nodes,
                    "expected_type": NodeView,
                    "expected_value": expected_nodes,
                },
            ]
        )
        # Checks for directed LabelGraph
        directed_graph = LabelGraph(directed=True)
        # Check for initiated directed LabelGraph
        CommonGraphMethods().check_graph_non_list_attributes(
            [
                {"attribute": directed_graph.directed, "expected_value": True},
                {"attribute": directed_graph.num_labels, "expected_value": 0},
                {"attribute": directed_graph.num_nodes(), "expected_value": 0},
                {"attribute": directed_graph.type, "expected_value": "graph"},
            ]
        )
        # Check for directed LabelGraph with added edges and nodes
        directed_graph.add_edges([edges.edge_0_to_0_1, edges.edge_0_1_to_0_2, edges.edge_0_2_to_0])
        CommonGraphMethods().check_graph_non_list_attributes(
            [
                {"attribute": directed_graph.directed, "expected_value": True},
                {"attribute": directed_graph.num_labels, "expected_value": 3},
                {"attribute": directed_graph.num_nodes(), "expected_value": 3},
                {"attribute": directed_graph.type, "expected_value": "graph"},
            ]
        )
        expected_nodes = [labels.label_0, labels.label_0_1, labels.label_0_2]
        CommonGraphMethods().check_graph_list_attributes(
            [
                {
                    "attribute": directed_graph.edges,
                    "expected_type": OutMultiEdgeDataView,
                    "expected_value": [
                        (labels.label_0, labels.label_0_1, 0, {}),
                        (labels.label_0_1, labels.label_0_2, 0, {}),
                        (labels.label_0_2, labels.label_0, 0, {}),
                    ],
                },
                {
                    "attribute": directed_graph.nodes,
                    "expected_type": NodeView,
                    "expected_value": expected_nodes,
                },
            ]
        )

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_graph_subgraph(self):
        """
        <b>Description:</b>
        Check "subgraph" method of LabelGraph class object

        <b>Input data:</b>
        LabelGraph objects with specified directed parameter and added edges

        <b>Expected results:</b>
        Test passes if subgraph method returns expected LabelGraph instance for LabelGraph object

        <b>Steps</b>
        1. Check LabelGraph instance returned by subgraph method for non-directed LabelGraph object
        2. Check LabelGraph instance returned by subgraph method for directed LabelGraph object
        """
        # Checks for non-directed LabelGraph
        non_directed_graph = self.non_directed_label_graph()
        # Checking subgraph with one node not included to parent graph
        non_directed_subgraph = non_directed_graph.subgraph(
            [labels.label_0_1, labels.label_0_2, labels.non_included_label]
        )
        CommonGraphMethods().check_graph_non_list_attributes(
            [
                {"attribute": non_directed_subgraph.directed, "expected_value": False},
                {"attribute": non_directed_subgraph.num_labels, "expected_value": 2},
                {"attribute": non_directed_subgraph.num_nodes(), "expected_value": 2},
                {"attribute": non_directed_subgraph.type, "expected_value": "graph"},
            ]
        )
        CommonGraphMethods().check_graph_list_attributes(
            [
                {
                    "attribute": non_directed_subgraph.edges,
                    "expected_type": EdgeDataView,
                    "expected_value": [(labels.label_0_1, labels.label_0_2, {})],
                },
                {
                    "attribute": non_directed_subgraph.nodes,
                    "expected_type": NodeView,
                    "expected_value": [labels.label_0_1, labels.label_0_2],
                },
            ]
        )

        # Checks for directed LabelGraph
        directed_graph = self.directed_label_graph()
        # Checking subgraph with one node not included to parent graph
        directed_subgraph = directed_graph.subgraph([labels.label_0_1, labels.label_0_2, labels.non_included_label])
        CommonGraphMethods().check_graph_non_list_attributes(
            [
                {"attribute": directed_subgraph.directed, "expected_value": True},
                {"attribute": directed_subgraph.num_labels, "expected_value": 2},
                {"attribute": directed_subgraph.num_nodes(), "expected_value": 2},
                {"attribute": directed_subgraph.type, "expected_value": "graph"},
            ]
        )
        CommonGraphMethods().check_graph_list_attributes(
            [
                {
                    "attribute": directed_subgraph.edges,
                    "expected_type": OutMultiEdgeDataView,
                    "expected_value": [(labels.label_0_1, labels.label_0_2, 0, {})],
                },
                {
                    "attribute": directed_subgraph.nodes,
                    "expected_type": NodeView,
                    "expected_value": [labels.label_0_1, labels.label_0_2],
                },
            ]
        )

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_graph_eq(self):
        """
        <b>Description:</b>
        Check __eq__ method of LabelGraph class object

        <b>Input data:</b>
        LabelGraph objects with specified directed parameter and added edges

        <b>Expected results:</b>
        Test passes if __eq__ method returns expected values

        <b>Steps</b>
        1. Check __eq__ method for non-directed LabelGraph object
        2. Check __eq__ method for directed LabelGraph object
        """
        # Checks for non-directed LabelGraph
        # Checking __eq__ method for equal non-directed LabelGraph
        non_directed_graph = self.non_directed_label_graph()
        equal_non_directed_graph = self.non_directed_label_graph()
        assert non_directed_graph == equal_non_directed_graph
        # Checking __eq__ method for equal non-directed and directed LabelGraph objects
        directed_graph = LabelGraph(directed=True)
        directed_graph.add_edges([edges.edge_0_to_0_1, edges.edge_0_1_to_0_2])
        assert non_directed_graph != directed_graph
        # Checking __eq__ method for non-directed LabelGraph objects with different edges
        non_directed_graph_different_edges = self.non_directed_label_graph()
        non_directed_graph_different_edges.add_edge(labels.label_0, labels.label_0_2)
        assert non_directed_graph != non_directed_graph_different_edges
        # Checking __eq__ method for non-directed LabelGraph objects with different nodes
        non_directed_graph_different_nodes = self.non_directed_label_graph()
        non_directed_graph_different_nodes.add_node(labels.non_included_label)
        assert non_directed_graph != non_directed_graph_different_nodes
        # Checking __eq__ method by comparing non-directed LabelGraph with different type object
        assert non_directed_graph != str
        # Checks for directed LabelGraph
        # Checking __eq__ method for equal directed LabelGraph
        directed_graph = self.directed_label_graph()
        equal_directed_graph = self.directed_label_graph()
        assert directed_graph == equal_directed_graph
        # Checking __eq__ method for directed LabelGraph objects with different edges
        directed_graph_different_edges = self.directed_label_graph()
        directed_graph_different_edges.add_edge(labels.label_0_1, labels.label_0)
        assert directed_graph != directed_graph_different_edges
        # Checking __eq__ method for directed LabelGraph objects with different nodes
        directed_graph_different_nodes = self.directed_label_graph()
        directed_graph_different_nodes.add_node(labels.non_included_label)
        assert directed_graph != directed_graph_different_nodes
        # Checking __eq__ method by comparing directed LabelGraph with different type object
        assert directed_graph != str


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestLabelTree:
    @staticmethod
    def label_tree_no_children() -> LabelTree:
        label_tree = LabelTree()
        label_tree.add_edges([edges.edge_0_to_0_1, edges.edge_0_to_0_2])
        return label_tree

    @staticmethod
    def label_tree() -> LabelTree:
        label_tree = LabelTree()
        label_tree.get_labels_in_topological_order()
        # Forming Label Tree with children
        for parent, child in [
            edges.edge_0_to_0_1,
            edges.edge_0_to_0_2,
            edges.edge_0_1_to_0_1_3,
            edges.edge_0_2_to_0_2_4,
            edges.edge_0_2_to_0_2_5,
        ]:
            label_tree.add_child(parent, child)
        return label_tree

    @staticmethod
    def check_get_children_method(label_tree) -> None:
        for label, expected in [
            (labels.label_0, [labels.label_0_1, labels.label_0_2]),
            (labels.label_0_1, [labels.label_0_1_3]),
            (labels.label_0_2, [labels.label_0_2_4, labels.label_0_2_5]),
            (labels.label_0_1_3, []),
            (labels.label_0_2_4, []),
            (labels.label_0_2_5, []),
            (LabelEntity("not included", domain=Domain.CLASSIFICATION), []),
        ]:
            assert label_tree.get_children(label) == expected

    @staticmethod
    def check_get_descendants_method(label_tree) -> None:
        for label, expected in [
            (
                labels.label_0,
                [
                    labels.label_0_1,
                    labels.label_0_1_3,
                    labels.label_0_2,
                    labels.label_0_2_4,
                    labels.label_0_2_5,
                ],
            ),
            (labels.label_0_1, [labels.label_0_1_3]),
            (labels.label_0_2, [labels.label_0_2_4, labels.label_0_2_5]),
            (labels.label_0_1_3, []),
            (labels.label_0_2_4, []),
            (labels.label_0_2_5, []),
        ]:
            assert label_tree.get_descendants(label) == expected

    @staticmethod
    def check_get_ancestors_method(label_tree) -> None:
        for label, expected in [
            (labels.label_0, [labels.label_0]),
            (labels.label_0_1, [labels.label_0_1, labels.label_0]),
            (labels.label_0_2, [labels.label_0_2, labels.label_0]),
            (
                labels.label_0_1_3,
                [labels.label_0_1_3, labels.label_0_1, labels.label_0],
            ),
            (
                labels.label_0_2_4,
                [labels.label_0_2_4, labels.label_0_2, labels.label_0],
            ),
            (
                labels.label_0_2_5,
                [labels.label_0_2_5, labels.label_0_2, labels.label_0],
            ),
        ]:
            assert label_tree.get_ancestors(label) == expected

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_tree_initialization(self):
        """
        <b>Description:</b>
        Check LabelTree class object initialization

        <b>Input data:</b>
        LabelTree object, edges and nodes to add

        <b>Expected results:</b>
        Test passes if LabelTree object "directed", "edges" and "nodes" attributes and "num_labels" and "type"
        properties and value returned by "num_nodes" method are equal expected
        """
        label_tree = LabelTree()
        # Check for initiated non-directed LabelGraph
        CommonGraphMethods().check_graph_non_list_attributes(
            [
                {"attribute": label_tree.directed, "expected_value": True},
                {"attribute": label_tree.num_labels, "expected_value": 0},
                {"attribute": label_tree.num_nodes(), "expected_value": 0},
                {"attribute": label_tree.type, "expected_value": "tree"},
            ]
        )
        # Check for LabelTree with added edges and nodes
        label_tree.add_edges([edges.edge_0_to_0_1, edges.edge_0_to_0_2])
        CommonGraphMethods().check_graph_non_list_attributes(
            [
                {"attribute": label_tree.directed, "expected_value": True},
                {"attribute": label_tree.num_labels, "expected_value": 3},
                {"attribute": label_tree.num_nodes(), "expected_value": 3},
                {"attribute": label_tree.type, "expected_value": "tree"},
            ]
        )
        expected_nodes = [labels.label_0, labels.label_0_1, labels.label_0_2]
        CommonGraphMethods().check_graph_list_attributes(
            [
                {
                    "attribute": label_tree.edges,
                    "expected_type": OutMultiEdgeDataView,
                    "expected_value": [
                        (labels.label_0, labels.label_0_1, 0, {}),
                        (labels.label_0, labels.label_0_2, 0, {}),
                    ],
                },
                {
                    "attribute": label_tree.nodes,
                    "expected_type": NodeView,
                    "expected_value": expected_nodes,
                },
            ]
        )

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_tree_add_edge(self):
        """
        <b>Description:</b>
        Check LabelTree class add_edge and add_edges methods

        <b>Input data:</b>
        LabelTree object with specified directed parameters and added edges

        <b>Expected results:</b>
        Test passes if "edges" attribute of LabelTree is equal expected value after using add_edge and add_edges methods
        """
        label_tree = self.label_tree_no_children()
        # Adding edges, one of which already in LabelTree
        label_tree.topological_sort()
        label_tree.add_edges([edges.edge_0_to_0_1, edges.edge_0_1_to_0_1_3])
        CommonGraphMethods().check_graph_non_list_attributes(
            [
                {"attribute": label_tree.num_labels, "expected_value": 4},
                {"attribute": label_tree.num_nodes(), "expected_value": 4},
            ]
        )
        CommonGraphMethods().check_graph_list_attributes(
            [
                {
                    "attribute": label_tree.edges,
                    "expected_type": OutMultiEdgeDataView,
                    "expected_value": [
                        (labels.label_0, labels.label_0_1, 0, {}),
                        (labels.label_0, labels.label_0_1, 1, {}),
                        (labels.label_0, labels.label_0_2, 0, {}),
                        (labels.label_0_1, labels.label_0_1_3, 0, {}),
                    ],
                },
                {
                    "attribute": label_tree.nodes,
                    "expected_type": NodeView,
                    "expected_value": [
                        labels.label_0,
                        labels.label_0_1,
                        labels.label_0_2,
                        labels.label_0_1_3,
                    ],
                },
            ]
        )
        assert not label_tree._LabelTree__topological_order_cache
        # Adding one existing and one non-existing edge
        label_tree.topological_sort()
        label_tree.add_edges([edges.edge_0_to_0_2, edges.edge_0_2_to_0_2_4])
        CommonGraphMethods().check_graph_non_list_attributes(
            [
                {"attribute": label_tree.num_labels, "expected_value": 5},
                {"attribute": label_tree.num_nodes(), "expected_value": 5},
            ]
        )
        CommonGraphMethods().check_graph_list_attributes(
            [
                {
                    "attribute": label_tree.edges,
                    "expected_type": OutMultiEdgeDataView,
                    "expected_value": [
                        (labels.label_0, labels.label_0_1, 0, {}),
                        (labels.label_0, labels.label_0_1, 1, {}),
                        (labels.label_0, labels.label_0_2, 0, {}),
                        (labels.label_0, labels.label_0_2, 1, {}),
                        (labels.label_0_1, labels.label_0_1_3, 0, {}),
                        (labels.label_0_2, labels.label_0_2_4, 0, {}),
                    ],
                },
                {
                    "attribute": label_tree.nodes,
                    "expected_type": NodeView,
                    "expected_value": [
                        labels.label_0,
                        labels.label_0_1,
                        labels.label_0_2,
                        labels.label_0_1_3,
                        labels.label_0_2_4,
                    ],
                },
            ]
        )
        assert not label_tree._LabelTree__topological_order_cache

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_tree_add_node(self):
        """
        <b>Description:</b>
        Check LabelTree class add_node method

        <b>Input data:</b>
        LabelTree object with specified directed parameter and added edges

        <b>Expected results:</b>
        Test passes if "nodes" attribute of LabelTree is equal expected value after using add_node method
        """
        label_tree = self.label_tree_no_children()
        # Adding new node
        label_tree.get_labels_in_topological_order()
        label_tree.add_node(labels.label_0_1_3)
        CommonGraphMethods().check_graph_non_list_attributes(
            [
                {"attribute": label_tree.num_labels, "expected_value": 4},
                {"attribute": label_tree.num_nodes(), "expected_value": 4},
            ]
        )
        expected_edges = [
            (labels.label_0, labels.label_0_1, 0, {}),
            (labels.label_0, labels.label_0_2, 0, {}),
        ]
        expected_nodes = [
            labels.label_0,
            labels.label_0_1,
            labels.label_0_2,
            labels.label_0_1_3,
        ]
        CommonGraphMethods().check_graph_list_attributes(
            [
                {
                    "attribute": label_tree.edges,
                    "expected_type": OutMultiEdgeDataView,
                    "expected_value": expected_edges,
                },
                {
                    "attribute": label_tree.nodes,
                    "expected_type": NodeView,
                    "expected_value": expected_nodes,
                },
            ]
        )
        assert not label_tree._LabelTree__topological_order_cache
        # Adding existing node, only topological_order_cache should be empty
        label_tree.get_labels_in_topological_order()
        label_tree.add_node(labels.label_0)
        CommonGraphMethods().check_graph_non_list_attributes(
            [
                {"attribute": label_tree.num_labels, "expected_value": 4},
                {"attribute": label_tree.num_nodes(), "expected_value": 4},
            ]
        )
        CommonGraphMethods().check_graph_list_attributes(
            [
                {
                    "attribute": label_tree.edges,
                    "expected_type": OutMultiEdgeDataView,
                    "expected_value": expected_edges,
                },
                {
                    "attribute": label_tree.nodes,
                    "expected_type": NodeView,
                    "expected_value": expected_nodes,
                },
            ]
        )
        assert not label_tree._LabelTree__topological_order_cache

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_tree_clear_topological_cache(self):
        """
        <b>Description:</b>
        Check LabelTree class clear_topological_cache method

        <b>Input data:</b>
        LabelTree object with specified directed parameter and added edges

        <b>Expected results:</b>
        Test passes if "__topological_order_cache" attribute of LabelTree is equal "None" after clear_topological_cache
        """
        # Check for empty LabelTree
        label_tree = LabelTree()
        label_tree.get_labels_in_topological_order()
        label_tree.clear_topological_cache()
        assert not label_tree._LabelTree__topological_order_cache
        # Check for LabelTree with specified nodes and edges
        label_tree = self.label_tree_no_children()
        label_tree.get_labels_in_topological_order()
        label_tree.clear_topological_cache()
        assert not label_tree._LabelTree__topological_order_cache

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_tree_relations(self):
        """
        <b>Description:</b>
        Check LabelTree class relations methods

        <b>Input data:</b>
        LabelTree object with specified directed parameter, added edges and children

        <b>Expected results:</b>
        Test passes if "get_parent", "get_children", "get_descendants", "get_siblings" and "get_ancestors" methods
        of LabelTree return expected values

        <b>Steps</b>
        1. Check add_children method
        2. Check "get_parent" method
        3. Check "get_descendants" method
        4. Check "get_siblings" method
        5. Check "get_ancestors" method
        """
        label_tree = self.label_tree()
        assert not label_tree._LabelTree__topological_order_cache
        # Checking new nodes and edges added after add_children method
        CommonGraphMethods().check_graph_non_list_attributes(
            [
                {"attribute": label_tree.num_labels, "expected_value": 6},
                {"attribute": label_tree.num_nodes(), "expected_value": 6},
            ]
        )
        CommonGraphMethods().check_graph_list_attributes(
            [
                {
                    "attribute": label_tree.edges,
                    "expected_type": OutMultiEdgeDataView,
                    "expected_value": [
                        (labels.label_0_1, labels.label_0, 0, {"value": None}),
                        (labels.label_0_2, labels.label_0, 0, {"value": None}),
                        (labels.label_0_1_3, labels.label_0_1, 0, {"value": None}),
                        (labels.label_0_2_4, labels.label_0_2, 0, {"value": None}),
                        (labels.label_0_2_5, labels.label_0_2, 0, {"value": None}),
                    ],
                },
                {
                    "attribute": label_tree.nodes,
                    "expected_type": NodeView,
                    "expected_value": [
                        labels.label_0_1,
                        labels.label_0,
                        labels.label_0_2,
                        labels.label_0_1_3,
                        labels.label_0_2_4,
                        labels.label_0_2_5,
                    ],
                },
            ]
        )
        # Checking "get_parent" method
        for label, expected in [
            (labels.label_0, None),
            (labels.label_0_1, labels.label_0),
            (labels.label_0_2, labels.label_0),
            (labels.label_0_1_3, labels.label_0_1),
            (labels.label_0_2_4, labels.label_0_2),
            (labels.label_0_2_5, labels.label_0_2),
        ]:
            assert label_tree.get_parent(label) == expected
        # Checking "get_children" method
        self.check_get_children_method(label_tree)
        # Checking "get_descendants" method
        self.check_get_descendants_method(label_tree)
        # Checking "get_siblings" method
        for label, expected in [
            (labels.label_0, []),
            (labels.label_0_1, [labels.label_0_2]),
            (labels.label_0_2, [labels.label_0_1]),
            (labels.label_0_1_3, []),
            (labels.label_0_2_4, [labels.label_0_2_5]),
            (labels.label_0_2_5, [labels.label_0_2_4]),
        ]:
            assert label_tree.get_siblings(label) == expected
        # Checking "get_ancestors" method
        self.check_get_ancestors_method(label_tree)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_tree_get_labels_in_topological_order(self):
        """
        <b>Description:</b>
        Check LabelTree class get_labels_in_topological_order method

        <b>Input data:</b>
        LabelTree object with specified directed parameter, added edges and children

        <b>Expected results:</b>
        Test passes if get_labels_in_topological_order method of LabelTree returns expected value

        <b>Steps</b>
        1. Check value returned by get_labels_in_topological_order method for Tree with multiple children branches
        2. Remove node with children from tree and check value returned by get_labels_in_topological_order method
        """
        label_tree = self.label_tree()
        # Checking value returned by get_labels_in_topological_order method for tree with multiple branches
        labels_topological_order = label_tree.get_labels_in_topological_order()

        def previous_vertexes(vert_id):
            vertexes = vert_id.split("_")
            return vertexes[:-1], vertexes[-1]

        returned_vertexes = set()
        for label in labels_topological_order:
            previous, curent = previous_vertexes(label.id_)
            for vertex in previous:
                assert vertex in returned_vertexes
            returned_vertexes.add(curent)
        assert {"0", "1", "2", "3", "4", "5"} == returned_vertexes

        assert label_tree._LabelTree__topological_order_cache == labels_topological_order
        # Removing node with children and checking value returned by get_labels_in_topological_order method
        label_tree.remove_node(labels.label_0_1)
        labels_topological_order = label_tree.get_labels_in_topological_order()
        returned_vertexes = set()
        for label in labels_topological_order:
            previous, curent = previous_vertexes(label.id_)
            for vertex in previous:
                if curent != "3":  # the '1' has been removed, so that '3' is separated from the rest graph.
                    assert vertex in returned_vertexes
            returned_vertexes.add(curent)
        assert {"0", "2", "3", "4", "5"} == returned_vertexes
        assert label_tree._LabelTree__topological_order_cache == labels_topological_order

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_tree_remove_node(self):
        """
        <b>Description:</b>
        Check LabelTree class remove_node method

        <b>Input data:</b>
        LabelTree object with specified directed parameter, added edges and children

        <b>Expected results:</b>
        Test passes if after using remove_node method on LabelTree object "edges", "nodes" and "num_labels" properties
        and "num_nodes" method return expected values

        <b>Steps</b>
        1. Check values returned by "edges", "nodes" and "num_labels" properties and "num_nodes" method after removing
        children node
        2. Check values returned by "edges", "nodes" and "num_labels" properties and "num_nodes" method after removing
        parent node
        """
        label_tree = self.label_tree()
        # Removing children node and checking "edges", "nodes" and "num_labels" properties and "num_nodes" method values
        label_tree.remove_node(labels.label_0_1_3)
        assert label_tree.num_nodes() == 5
        assert label_tree.num_labels == 5
        assert not label_tree._LabelTree__topological_order_cache
        CommonGraphMethods().check_graph_list_attributes(
            [
                {
                    "attribute": label_tree.edges,
                    "expected_type": OutMultiEdgeDataView,
                    "expected_value": [
                        (labels.label_0_1, labels.label_0, 0, {"value": None}),
                        (labels.label_0_2, labels.label_0, 0, {"value": None}),
                        (labels.label_0_2_4, labels.label_0_2, 0, {"value": None}),
                        (labels.label_0_2_5, labels.label_0_2, 0, {"value": None}),
                    ],
                },
                {
                    "attribute": label_tree.nodes,
                    "expected_type": NodeView,
                    "expected_value": [
                        labels.label_0_1,
                        labels.label_0,
                        labels.label_0_2,
                        labels.label_0_2_4,
                        labels.label_0_2_5,
                    ],
                },
            ]
        )
        # Removing node with children and checking value returned by get_labels_in_topological_order method
        label_tree.get_labels_in_topological_order()
        label_tree.remove_node(labels.label_0_2)
        assert label_tree.num_nodes() == 4
        assert label_tree.num_labels == 4
        assert not label_tree._LabelTree__topological_order_cache
        expected_edges = [(labels.label_0_1, labels.label_0, 0, {"value": None})]
        CommonGraphMethods().check_graph_list_attributes(
            [
                {
                    "attribute": label_tree.edges,
                    "expected_type": OutMultiEdgeDataView,
                    "expected_value": expected_edges,
                },
                {
                    "attribute": label_tree.nodes,
                    "expected_type": NodeView,
                    "expected_value": [
                        labels.label_0_1,
                        labels.label_0,
                        labels.label_0_2_4,
                        labels.label_0_2_5,
                    ],
                },
            ]
        )

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_tree_subgraph(self):
        """
        <b>Description:</b>
        Check LabelTree class subgraph method

        <b>Input data:</b>
        LabelTree object with specified directed parameter, added edges and children

        <b>Expected results:</b>
        Test passes if LabelTree object returned by subgraph method is equal expected
        """
        label_tree = self.label_tree()
        non_included_label = LabelEntity("not included", domain=Domain.CLASSIFICATION)
        subgraph = label_tree.subgraph(
            [
                labels.label_0,
                labels.label_0_1,
                labels.label_0_2,
                labels.label_0_2_5,
                non_included_label,
            ]
        )
        assert subgraph.num_nodes() == 4
        assert subgraph.num_labels == 4
        CommonGraphMethods().check_graph_list_attributes(
            [
                {
                    "attribute": subgraph.edges,
                    "expected_type": OutMultiEdgeDataView,
                    "expected_value": [
                        (labels.label_0_1, labels.label_0, 0, {"value": None}),
                        (labels.label_0_2, labels.label_0, 0, {"value": None}),
                        (labels.label_0_2_5, labels.label_0_2, 0, {"value": None}),
                    ],
                },
                {
                    "attribute": subgraph.nodes,
                    "expected_type": NodeView,
                    "expected_value": [
                        labels.label_0_1,
                        labels.label_0,
                        labels.label_0_2,
                        labels.label_0_2_5,
                    ],
                },
            ]
        )

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_tree_eq(self):
        """
        <b>Description:</b>
        Check LabelTree class __eq__ method

        <b>Input data:</b>
        LabelTree objects with specified directed parameter, added edges and children

        <b>Expected results:</b>
        Test passes if value returned by __eq__ method is equal expected

        <b>Steps</b>
        1. Check value returned by __eq__ method for equal LabelTree objects
        2. Check value returned by __eq__ method for LabelTree objects with different edges
        3. Check value returned by __eq__ method for LabelTree objects with different nodes
        4. Check value returned by __eq__ method for comparing LabelTree objects with different type object
        """
        label_tree = self.label_tree()
        # Checking __eq__ method for equal LabelTree objects
        equal_label_tree = self.label_tree()
        assert label_tree == equal_label_tree
        # Checking __eq__ method for LabelTree objects with different edges
        different_edges_tree = self.label_tree()
        different_edges_tree.add_edge(labels.label_0, labels.label_0_1)
        assert not label_tree == different_edges_tree
        # Checking __eq__ method for LabelTree objects with different nodes
        different_nodes_tree = self.label_tree()
        different_nodes_tree.add_node(LabelEntity("not included", domain=Domain.CLASSIFICATION))
        assert label_tree != different_nodes_tree
        # Checking __eq__ method for comparing LabelTree object with different type object
        assert label_tree != str


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestLabelSchemaEntity:
    @staticmethod
    def label_groups() -> list:
        group_1 = LabelGroup(
            name="Exclusive group 1",
            labels=[labels.label_0_1, labels.label_0_2],
            id=ID("Exclusive group 1"),
        )
        group_2 = LabelGroup(
            name="Exclusive group 2",
            labels=[labels.label_0_2_4, labels.label_0_2_5],
            id=ID("Exclusive group 2"),
        )
        return [group_1, group_2]

    @staticmethod
    def empty_labels() -> list:
        empty_label = LabelEntity(
            name="Empty label",
            domain=Domain.SEGMENTATION,
            color=Color(255, 255, 255),
            is_empty=True,
            id=ID("empty_label_1"),
        )
        empty_non_exclusive_label = LabelEntity(
            name="Empty non-exclusive label",
            domain=Domain.DETECTION,
            color=Color(255, 255, 255),
            is_empty=True,
            id=ID("empty_non_excl_label_1"),
        )
        return [empty_label, empty_non_exclusive_label]

    def empty_labels_groups(self) -> list:
        empty_labels = self.empty_labels()
        empty_group_1 = LabelGroup(
            name="Exclusive group with empty label",
            labels=[empty_labels[0]],
            id=ID("Exclusive group with empty label"),
        )
        empty_group_2 = LabelGroup(
            name="Empty label group with empty label",
            labels=[empty_labels[1]],
            group_type=LabelGroupType.EMPTY_LABEL,
            id=ID("Empty label group with empty label"),
        )
        return [empty_group_1, empty_group_2]

    def label_schema_entity(self) -> LabelSchemaEntity:
        return LabelSchemaEntity(
            label_tree=TestLabelTree.label_tree(),
            label_groups=self.label_groups() + self.empty_labels_groups(),
        )

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_schema_entity_default_parameters_initialization(self):
        """
        <b>Description:</b>
        Check LabelSchemaEntity object initialization

        <b>Input data:</b>
        LabelSchemaEntity object with not specified label_tree and label_groups parameters

        <b>Expected results:</b>
        Test passes if attributes for LabelSchemaEntity with not specified "label_tree" and
        "label_groups" parameters are equal expected
        """
        empty_label_schema_entity = LabelSchemaEntity()
        assert empty_label_schema_entity.label_tree == LabelTree()
        assert empty_label_schema_entity._groups == []

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_schema_entity_get_labels(self):
        """
        <b>Description:</b>
        Check LabelSchemaEntity class get_labels method

        <b>Input data:</b>
        LabelSchemaEntity object with specified label_tree and label_groups parameters

        <b>Expected results:</b>
        Test passes if list returned by get_labels method is equal expected

        <b>Steps</b>
        1. Check list returned by get_labels method with include_empty parameter set to True
        2. Check list returned by get_labels method with include_empty parameter set to False
        """
        empty_labels = self.empty_labels()
        label_schema_entity = self.label_schema_entity()
        # Checking list returned by get_labels method with include_empty parameter set to True
        assert label_schema_entity.get_labels(include_empty=True) == [
            labels.label_0_1,
            labels.label_0_2,
            labels.label_0_2_4,
            labels.label_0_2_5,
            empty_labels[0],
            empty_labels[1],
        ]
        # Checking list returned by get_labels method with include_empty parameter set to False
        assert label_schema_entity.get_labels(include_empty=False) == [
            labels.label_0_1,
            labels.label_0_2,
            labels.label_0_2_4,
            labels.label_0_2_5,
        ]

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_schema_entity_get_groups(self):
        """
        <b>Description:</b>
        Check LabelSchemaEntity class get_groups method

        <b>Input data:</b>
        LabelSchemaEntity object with specified label_tree and label_groups parameters

        <b>Expected results:</b>
        Test passes if list returned by get_groups method is equal expected

        <b>Steps</b>
        1. Check list returned by get_groups method with include_empty parameter set to True
        2. Check list returned by get_groups method with include_empty parameter set to False
        """
        label_groups = self.label_groups()
        empty_label_groups = self.empty_labels_groups()
        label_schema_entity = self.label_schema_entity()
        # Checking list returned by get_groups method with include_empty parameter set to True
        assert label_schema_entity.get_groups(include_empty=True) == [
            label_groups[0],
            label_groups[1],
            empty_label_groups[0],
            empty_label_groups[1],
        ]
        # Checking list returned by get_groups method with include_empty parameter set to False
        assert label_schema_entity.get_groups(include_empty=False) == [
            label_groups[0],
            label_groups[1],
            empty_label_groups[0],
        ]

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_schema_entity_add_group(self):
        """
        <b>Description:</b>
        Check LabelSchemaEntity class add_group method

        <b>Input data:</b>
        LabelSchemaEntity object with label_tree and label_groups parameters

        <b>Expected results:</b>
        Test passes if value returned by "get_exclusive_groups" method is equal expected

        <b>Steps</b>
        1. Check value returned by "get_exclusive_groups" method after adding group with new labels
        2. Check value returned by "get_exclusive_groups" method after adding group with single label
        3. Check value returned by "get_exclusive_groups" method after adding group with already added label
        4. Check value returned by "get_exclusive_groups" method and exclusivity of labels after adding group exclusive
        to other
        5. Check value returned by "get_exclusive_groups" method after adding non-exclusive group
        6. Check LabelGroupExistsException raised when adding LabelGroup with already existing name
        """
        empty_label_groups = self.empty_labels_groups()
        exclusive_groups = self.label_groups() + [empty_label_groups[0]]
        label_schema_entity = self.label_schema_entity()
        # Scenario for adding exclusive group with new labels
        new_exclusive_label = LabelEntity(name="New label", domain=Domain.DETECTION, id=ID("new_ex_1"))
        other_new_exclusive_label = LabelEntity(name="Other new label", domain=Domain.DETECTION, id=ID("new_ex_2"))
        new_exclusive_group = LabelGroup(
            name="New exclusive labels group",
            labels=[new_exclusive_label, other_new_exclusive_label],
            id=ID("new_ex_group"),
        )
        label_schema_entity.add_group(new_exclusive_group)
        assert label_schema_entity.get_exclusive_groups() == (exclusive_groups + [new_exclusive_group])
        # Scenario for adding exclusive group with single label
        label_schema_entity = self.label_schema_entity()
        new_exclusive_group = LabelGroup(
            name="Exclusive group with one label",
            labels=[new_exclusive_label],
            id=ID("single_excl_group"),
        )
        label_schema_entity.add_group(new_exclusive_group)
        assert label_schema_entity.get_exclusive_groups() == (exclusive_groups + [new_exclusive_group])
        # Scenario for adding exclusive group with one already existing label
        label_schema_entity = self.label_schema_entity()
        new_exclusive_group = LabelGroup(
            name="Exclusive group to link with existing",
            labels=[labels.label_0_1, new_exclusive_label],
            id=ID("new_ex_group"),
        )
        label_schema_entity.add_group(new_exclusive_group)
        assert label_schema_entity.get_exclusive_groups() == (exclusive_groups + [new_exclusive_group])
        # Scenario for adding non-exclusive group
        label_schema_entity = self.label_schema_entity()
        new_exclusive_group = LabelGroup(
            name="Non exclusive label group",
            labels=[new_exclusive_label],
            group_type=LabelGroupType.EMPTY_LABEL,
            id=ID("non_exclusive_group"),
        )
        label_schema_entity.add_group(new_exclusive_group)
        assert label_schema_entity.get_exclusive_groups() == exclusive_groups
        # Raise LabelGroupExistsException when adding LabelGroup with same name
        for group_type in [LabelGroupType.EXCLUSIVE, LabelGroupType.EMPTY_LABEL]:
            with pytest.raises(LabelGroupExistsException):
                label_schema_entity.add_group(
                    LabelGroup(
                        name="Exclusive group 1",
                        labels=[],
                        group_type=group_type,
                    )
                )

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_schema_add_child(self):
        """
        <b>Description:</b>
        Check LabelSchemaEntity class add_child method

        <b>Input data:</b>
        LabelSchemaEntity object with specified label_tree and label_groups parameters

        <b>Expected results:</b>
        Test passes if list returned by get_children method after using add_child method are equal to expected

        <b>Steps</b>
        1. Check get_children list after using add_child method
        2. Check get_children list after using add_child method for previous pair for a second time
        """
        label_schema_entity = self.label_schema_entity()
        label_schema_entity.add_child(parent=labels.label_0_2_4, child=labels.label_0_2_5)
        assert label_schema_entity.get_children(labels.label_0_2_4) == [labels.label_0_2_5]
        label_schema_entity.add_child(parent=labels.label_0_2_4, child=labels.label_0_2_5)
        assert label_schema_entity.get_children(labels.label_0_2_4) == [labels.label_0_2_5]

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_schema_entity_get_label_ids(self):
        """
        <b>Description:</b>
        Check LabelSchemaEntity class get_label_ids method

        <b>Input data:</b>
        LabelSchemaEntity object with specified label_tree and label_groups parameters

        <b>Expected results:</b>
        Test passes if values returned by get_label_ids method is equal expected

        <b>Steps</b>
        Check value returned by get_label_ids method for LabelSchemaEntity object
        """
        expected_non_empty_labels = [ID("0_1"), ID("0_2"), ID("0_2_4"), ID("0_2_5")]
        expected_include_empty_labels = expected_non_empty_labels + [
            ID("empty_label_1"),
            ID("empty_non_excl_label_1"),
        ]
        label_schema_entity = self.label_schema_entity()
        assert label_schema_entity.get_label_ids(include_empty=True) == expected_include_empty_labels
        assert label_schema_entity.get_label_ids(include_empty=False) == expected_non_empty_labels

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_schema_get_label_group_by_name(self):
        """
        <b>Description:</b>
        Check LabelSchemaEntity class get_label_group_by_name method

        <b>Input data:</b>
        LabelSchemaEntity object with specified label_tree and label_groups parameters

        <b>Expected results:</b>
        Test passes if value returned by get_label_group_by_name method is equal expected

        <b>Steps</b>
        1. Check get_label_group_by_name method for searching exclusive group
        2. Check get_label_group_by_name method for searching empty label group
        2. Check get_label_group_by_name method for searching non_existing group
        """
        label_schema_entity = self.label_schema_entity()
        label_groups = self.label_groups()
        empty_non_excl_group = self.empty_labels_groups()[1]
        # Checking get_label_group_by_name method for searching exclusive group for not specified empty labels
        assert label_schema_entity.get_label_group_by_name("Exclusive group 1") == label_groups[0]
        # Checking get_label_group_by_name method for searching empty label group with empty label
        assert label_schema_entity.get_label_group_by_name("Empty label group with empty label") == empty_non_excl_group
        # Checking get_label_group_by_name method for searching non-existing group
        assert not label_schema_entity.get_label_group_by_name("Non-existing group")

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_schema_get_exclusive_groups(self):
        """
        <b>Description:</b>
        Check LabelSchemaEntity class get_exclusive_groups method

        <b>Input data:</b>
        LabelSchemaEntity object with specified label_tree and label_groups parameters

        <b>Expected results:</b>
        Test passes if value returned by get_exclusive_groups method is equal expected

        <b>Steps</b>
        1. Check get_exclusive_groups method
        2. Check get_exclusive_groups method after adding exclusive group
        3. Check get_exclusive_groups method after adding empty label group
        """
        label_schema_entity = self.label_schema_entity()
        exclusive_groups = self.label_groups() + [self.empty_labels_groups()[0]]
        # Checking get_exclusive_groups method for searching exclusive groups
        assert label_schema_entity.get_exclusive_groups() == exclusive_groups
        # Checking get_exclusive_groups method after adding new exclusive group
        new_label = LabelEntity(
            name="New label",
            domain=Domain.DETECTION,
            color=Color(100, 16, 25),
            id=ID("new_ex_1"),
        )
        new_labels_group = LabelGroup(
            name="New exclusive labels group",
            labels=[new_label],
            id=ID("new_ex_group"),
        )
        label_schema_entity.add_group(new_labels_group)
        exclusive_groups.append(new_labels_group)
        assert label_schema_entity.get_exclusive_groups() == exclusive_groups
        # Checking get_exclusive_groups method after adding empty label group
        empty_label_group = LabelGroup(
            name="New non-exclusive labels group",
            labels=[new_label],
            group_type=LabelGroupType.EMPTY_LABEL,
            id=ID("new_ex_group"),
        )
        label_schema_entity.add_group(empty_label_group)
        assert label_schema_entity.get_exclusive_groups() == exclusive_groups

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_schema_add_labels_to_group_by_group_name(self):
        """
        <b>Description:</b>
        Check LabelSchemaEntity class add_labels_to_group_by_group_name method

        <b>Input data:</b>
        LabelSchemaEntity object with specified label_tree and label_groups parameters

        <b>Expected results:</b>
        Test passes if labels attribute returned by group which specified in add_labels_to_group_by_group_name method
        is equal expected

        <b>Steps</b>
        1. Check add_labels_to_group_by_group_name method to add labels to exclusive group
        2. Check add_labels_to_group_by_group_name method to add labels to empty label group
        3. Check LabelGroupDoesNotExistException raised when adding labels to non-existing group
        """
        label_schema_entity = self.label_schema_entity()
        non_exclusive_label = self.empty_labels()[1]
        # Checking add_labels_to_group_by_group_name method to add labels to exclusive group
        new_label = LabelEntity(
            name="New label",
            domain=Domain.DETECTION,
            color=Color(100, 16, 25),
            id=ID("new_ex_1"),
        )
        new_empty_label = LabelEntity(
            name="New empty label",
            domain=Domain.DETECTION,
            color=Color(81, 100, 10),
            id=ID("new_ex_2"),
        )
        exclusive_group_name = "Exclusive group 1"
        label_schema_entity.add_labels_to_group_by_group_name(
            group_name=exclusive_group_name, labels=[new_label, new_empty_label]
        )
        assert label_schema_entity.get_label_group_by_name(exclusive_group_name).labels == [
            labels.label_0_1,
            labels.label_0_2,
            new_label,
            new_empty_label,
        ]
        # Checking add_labels_to_group_by_group_name method to add labels to non-exclusive group
        new_empty_label = LabelEntity(
            name="New non-exclusive empty_label",
            domain=Domain.SEGMENTATION,
            is_empty=True,
            id=ID("empty_label_1"),
        )
        empty_label_group_name = "Empty label group with empty label"
        label_schema_entity.add_labels_to_group_by_group_name(
            group_name=empty_label_group_name, labels=[new_empty_label]
        )
        assert label_schema_entity.get_label_group_by_name(empty_label_group_name).labels == [
            non_exclusive_label,
            new_empty_label,
        ]
        # Checking that LabelGroupDoesNotExistException raised when adding labels to non-existing group
        with pytest.raises(LabelGroupDoesNotExistException):
            label_schema_entity.add_labels_to_group_by_group_name(
                group_name="Non-existing group", labels=[new_empty_label]
            )

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_schema_relations(self):
        """
        <b>Description:</b>
        Check LabelSchemaEntity relations methods

        <b>Input data:</b>
        LabelSchemaEntity object with specified label_tree and label_groups parameters

        <b>Expected results:</b>
        Test passes if "get_children", "get_descendants" and "get_ancestors" methods
        of LabelTree return expected values

        <b>Steps</b>
        1. Check "get_children" method
        2. Check "get_descendants" method
        3. Check "get_ancestors" method
        """
        label_schema_entity = self.label_schema_entity()
        # Checking get_children method
        TestLabelTree.check_get_children_method(label_schema_entity)
        # Checking get_descendants method
        TestLabelTree.check_get_descendants_method(label_schema_entity)
        # Checking get_ancestors method
        TestLabelTree.check_get_ancestors_method(label_schema_entity)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_schema_get_group_containing_label(self):
        """
        <b>Description:</b>
        Check LabelSchemaEntity class get_group_containing_label method

        <b>Input data:</b>
        LabelSchemaEntity object with specified label_tree and label_groups parameters

        <b>Expected results:</b>
        Test passes if value returned by get_group_containing_label method is equal expected

        <b>Steps</b>
        1. Check get_group_containing_label method for label included in exclusive group
        2. Check get_group_containing_label method for label included in empty label
        3. Check get_group_containing_label method for label not included in any group
        """
        label_schema_entity = self.label_schema_entity()
        label_groups = self.label_groups()
        # Checking get_group_containing_label method for label included in exclusive group
        assert label_schema_entity.get_group_containing_label(labels.label_0_1) == label_groups[0]
        assert label_schema_entity.get_group_containing_label(labels.label_0_2) == label_groups[0]
        assert label_schema_entity.get_group_containing_label(labels.label_0_2_4) == label_groups[1]
        assert label_schema_entity.get_group_containing_label(labels.label_0_2_5) == label_groups[1]
        # Checking get_group_containing_label method for label included in non-exclusive group
        assert label_schema_entity.get_group_containing_label(self.empty_labels()[1]) == self.empty_labels_groups()[1]
        # Checking get_group_containing_label method for label not included in any group
        assert not label_schema_entity.get_group_containing_label(labels.label_0)
        assert not label_schema_entity.get_group_containing_label(labels.label_0_1_3)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_schema_get_label(self):
        """
        <b>Description:</b>
        Check LabelSchemaEntity class __get_label method

        <b>Input data:</b>
        LabelSchemaEntity object with specified label_tree and label_groups parameters

        <b>Expected results:</b>
        Test passes if LabelEntity object returned by __get_label method is equal expected

        <b>Steps</b>
        1. Check __get_label method for searching ScoredLabel object
        2. Check __get_label method for searching LabelEntity object
        3. Check __get_label method for searching non-label object
        """
        label_schema_entity = self.label_schema_entity()
        # Checking __get_label for searching ScoredLabel object
        label_to_set_scored = LabelEntity(
            name="Scored label",
            domain=Domain.DETECTION,
            color=Color(red=100, green=50, blue=200),
            id=ID("scored_label_1"),
        )
        scored_label = ScoredLabel(label=label_to_set_scored)
        scored_labels_group = LabelGroup(name="Group with scored label", labels=[scored_label])
        label_schema_entity.add_group(scored_labels_group)
        assert label_schema_entity._LabelSchemaEntity__get_label(scored_label) == scored_label.get_label()
        # Checking __get_label for searching LabelEntity object
        assert label_schema_entity._LabelSchemaEntity__get_label(labels.label_0) == labels.label_0
        # Checking __get_label method for searching non-label object
        with pytest.raises(ValueError):
            label_schema_entity._LabelSchemaEntity__get_label(str)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_schema_repr(self):
        """
        <b>Description:</b>
        Check LabelSchemaEntity class __repr__ method

        <b>Input data:</b>
        LabelSchemaEntity object with specified label_tree and label_groups parameters

        <b>Expected results:</b>
        Test passes if value returned by __repr__ method is equal expected
        """
        label_schema_entity = self.label_schema_entity()
        label_groups = self.label_groups() + self.empty_labels_groups()
        assert repr(label_schema_entity) == f"LabelSchemaEntity(label_groups={label_groups})"

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_schema_eq(self):
        """
        <b>Description:</b>
        Check LabelSchemaEntity class __eq__ method

        <b>Input data:</b>
        LabelSchemaEntity objects with specified label_tree and label_groups parameters

        <b>Expected results:</b>
        Test passes if value returned by __eq__ method is equal expected

        <b>Steps</b>
        1. Check __eq__ method for equal LabelSchemaEntity objects
        2. Check __eq__ method for LabelSchemaEntity objects with unequal label_tree
        3. Check __eq__ method for LabelSchemaEntity objects with unequal LabelGroups
        4. Check __eq__ method for comparing LabelSchemaEntity object object of other type
        """
        # Checking __eq__ method for equal LabelSchemaEntity objects
        label_schema_entity = self.label_schema_entity()
        equal_label_schema_entity = self.label_schema_entity()
        assert label_schema_entity == equal_label_schema_entity
        # Checking __eq__ method for equal LabelSchemaEntity objects with unequal label_tree
        unequal_tree_label_schema_entity = self.label_schema_entity()
        unequal_tree_label_schema_entity.label_tree.add_edge(labels.label_0_1_3, labels.label_0_2_4)
        assert label_schema_entity != unequal_tree_label_schema_entity
        # Checking __eq__ method for equal LabelSchemaEntity objects with unequal LabelGroups
        unequal_groups_label_schema_entity = self.label_schema_entity()
        unequal_groups_label_schema_entity.add_labels_to_group_by_group_name("Exclusive group 1", [labels.label_0])
        assert label_schema_entity != unequal_groups_label_schema_entity
        # Checking __eq__ method for comparing LabelSchemaEntity object object of other type
        assert label_schema_entity != str

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_schema_from_labels(self):
        """
        <b>Description:</b>
        Check LabelSchemaEntity class from_labels method

        <b>Input data:</b>
        LabelSchemaEntity objects with specified label_tree and label_groups parameters

        <b>Expected results:</b>
        Test passes if LabelSchemaEntity object returned by from_labels method is equal expected
        """
        expected_labels = [
            labels.label_0,
            labels.label_0_1,
            labels.label_0_2,
            labels.label_0_2_4,
        ]
        label_schema_entity = LabelSchemaEntity.from_labels(expected_labels)
        labels_schema_entity_groups = label_schema_entity._groups
        assert isinstance(label_schema_entity, LabelSchemaEntity)
        assert label_schema_entity.label_tree == LabelTree()
        assert len(labels_schema_entity_groups) == 1
        assert labels_schema_entity_groups[0].name == "from_label_list"
        assert labels_schema_entity_groups[0].labels == expected_labels

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_schema_resolve_labels_probabilistic(self):
        """
        <b>Description:</b>
        Check LabelSchemaEntity class resolve_labels_probabilistic method

        <b>Input data:</b>
        LabelSchemaEntity objects with specified label_tree and label_groups parameters

        <b>Expected results:</b>
        Test passes if LabelSchemaEntity object returned by resolve_labels_probabilistic
        method is equal expected
        """
        label_schema = LabelSchemaEntity()
        labels_1 = [
            LabelEntity(
                name=f"Label {i}",
                domain=Domain.CLASSIFICATION,
                id=ID(f"{i}"),
                color=Color(i, 200, 166),
            )
            for i in range(2)
        ]
        labels_2 = [
            LabelEntity(
                name=f"Label {i}",
                domain=Domain.CLASSIFICATION,
                id=ID(f"{i}"),
                color=Color(i, 200, 166),
            )
            for i in range(2, 4)
        ]
        labels_3 = [
            LabelEntity(
                name=f"Label {i}",
                domain=Domain.CLASSIFICATION,
                id=ID(f"{i}"),
                color=Color(i, 200, 166),
            )
            for i in range(4, 5)
        ]
        group_1 = LabelGroup(name="labels1", labels=labels_1, group_type=LabelGroupType.EXCLUSIVE)
        label_schema.add_group(group_1)
        group_2 = LabelGroup(name="labels2", labels=labels_2, group_type=LabelGroupType.EXCLUSIVE)
        label_schema.add_group(group_2)

        label_schema.add_group(LabelGroup("child_labels_2", labels_3, LabelGroupType.EXCLUSIVE))
        label_schema.add_child(labels_2[0], labels_3[0])

        assert 1 == len(label_schema.get_descendants(labels_2[0]))

        # supress non-maximum labels
        predicted_labels = [
            ScoredLabel(labels_1[0], 0.1),
            ScoredLabel(labels_1[1], 0.5),
            ScoredLabel(labels_2[0], 0.2),
            ScoredLabel(labels_2[1], 0.5),
        ]
        resloved_labels = label_schema.resolve_labels_probabilistic(predicted_labels)
        assert [
            ScoredLabel(labels_1[1], 0.5),
            ScoredLabel(labels_2[1], 0.5),
        ] == resloved_labels

        # supress children of non-maximum labels
        predicted_labels = [
            ScoredLabel(labels_2[0], 0.1),
            ScoredLabel(labels_2[1], 0.5),
            ScoredLabel(labels_3[0], 0.4),
        ]
        resloved_labels = label_schema.resolve_labels_probabilistic(predicted_labels)
        assert [ScoredLabel(labels_2[1], 0.5)] == resloved_labels
