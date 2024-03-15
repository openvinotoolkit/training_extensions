# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from unittest.mock import patch

import pytest

from otx.api.entities.interfaces.graph_interface import IGraph
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestIGraph:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    @patch("otx.api.entities.interfaces.graph_interface.IGraph.__abstractmethods__", set())
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
