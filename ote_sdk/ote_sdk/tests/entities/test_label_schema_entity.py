"""This module tests classes related to LabelSchema"""

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

import pytest

from ote_sdk.entities.label_schema import (
    LabelGroup,
    LabelGroupType,
    LabelSchemaEntity,
    LabelTree,
)
from ote_sdk.tests.constants.ote_sdk_components import OteSdkComponent
from ote_sdk.tests.constants.requirements import Requirements


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestLabelSchema:
    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_tree(self, label_schema_example):
        """
        <b>Description:</b>
        Check that childs and parents of Labels can be found correctly

        <b>Input data:</b>
        A simple Label Tree

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
        assert rifle in threat_children

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_exclusive(self, label_schema_example):
        """
        <b>Description:</b>
        Tests exclusivity

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
                LabelGroupType.EXCLUSIVE,
            )
        )

        assert label_schema.are_exclusive(flowering, no_plant)
        assert label_schema.are_exclusive(vegetative, no_plant)
        assert label_schema.are_exclusive(vegetative, flowering)

        bee = label_schema_example.new_label_by_name("bee")
        label_schema.add_group(LabelGroup("bee_state", [bee], LabelGroupType.EXCLUSIVE))

        assert not label_schema.are_exclusive(flowering, bee)

        assert 4 == label_schema.exclusivity_graph.num_labels

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_complex(self, label_schema_example):
        """
        <b>Description:</b>
        Tests a mix of hierarchy and exclusivity

        <b>Input data:</b>
        Exclusive group: flowering,vegetative,no_plant
        Exclusive group: flower_partial_visible,flower_fully_visible (children of flowering)
        Exclusive group: few_leaves (child of vegetative)
        Exclusive group: bee

        <b>Expected results:</b>
        Test passes if the LabelSchema and hierarchy are correctly represented

        <b>Steps</b>
        1. Create LabelSchema with hierarchy
        2. Check that some relations are not exclusive
        3. Check that requesting scores of related labels does resolve correctly
        """
        label_schema = LabelSchemaEntity()
        label_schema_example.add_hierarchy(label_schema)

        bee = label_schema_example.new_label_by_name("bee")  # indicates presence/absence of bee
        label_schema.add_group(LabelGroup("bee_state", [bee], LabelGroupType.EXCLUSIVE))

        pollen_visible = label_schema_example.new_label_by_name("pollen_visible")
        queen = label_schema_example.new_label_by_name("queen")
        label_schema.add_child(bee, pollen_visible)
        label_schema.add_child(bee, queen)

        assert not label_schema.are_exclusive(pollen_visible, queen)
        assert not label_schema.are_exclusive(bee, queen)

    @pytest.mark.priority_medium
    @pytest.mark.component
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
    @pytest.mark.component
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
        1. Create LabelSchema with "Vegetation" exclusive group
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
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_labelschema_equality(self, label_schema_example):
        """
        <b>Description:</b>
        Check that LabelSchema equality works correctly

        <b>Input data:</b>
        LabelSchema instances

        <b>Expected results:</b>
        == and != operations work correctly for various inputs

        <b>Steps</b>
        1. Test LabelSchema equality
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
