# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from unittest.mock import patch

import pytest

from ote_sdk.entities.interfaces.graph_interface import IGraph
from ote_sdk.tests.constants.ote_sdk_components import OteSdkComponent
from ote_sdk.tests.constants.requirements import Requirements


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestIGraph:
    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    @patch(
        "ote_sdk.entities.interfaces.graph_interface.IGraph.__abstractmethods__", set()
    )
    def test_i_graph(self):
        """
        <b>Description:</b>
        Check IGraph class object initialization

        <b>Input data:</b>
        IGraph object

        <b>Expected results:</b>
        Test passes if IGraph object methods raise NotImplementedError exception
        """
        i_graph = IGraph()
        with pytest.raises(NotImplementedError):
            i_graph.add_node(1)
        with pytest.raises(NotImplementedError):
            i_graph.add_edge(1, 2)
        with pytest.raises(NotImplementedError):
            i_graph.has_edge_between(1, 2)
        with pytest.raises(NotImplementedError):
            i_graph.neighbors(1)
        with pytest.raises(NotImplementedError):
            i_graph.find_cliques()
        with pytest.raises(NotImplementedError):
            i_graph.num_nodes()
        with pytest.raises(NotImplementedError):
            i_graph.remove_edges(1, 2)
        with pytest.raises(NotImplementedError):
            i_graph.find_out_edges(1)
        with pytest.raises(NotImplementedError):
            i_graph.find_in_edges(1)
        with pytest.raises(NotImplementedError):
            i_graph.edges
        with pytest.raises(NotImplementedError):
            i_graph.nodes
