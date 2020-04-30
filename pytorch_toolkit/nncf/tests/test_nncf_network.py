"""
 Copyright (c) 2019-2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import itertools
from typing import List

import networkx as nx
import pytest
import torch
from copy import deepcopy
from torch import nn

from nncf.dynamic_graph.context import Scope
from nncf.dynamic_graph.graph import InputAgnosticOperationExecutionContext, NNCFGraph, OperationExecutionContext
from nncf.dynamic_graph.graph_builder import ModelInputInfo
from nncf.dynamic_graph.operator_metatypes import NoopMetatype
from nncf.dynamic_graph.patch_pytorch import MODEL_INPUT_OP_NAME
from nncf.module_operations import BaseOp
from nncf.nncf_network import NNCFNetwork, InsertionCommand, InsertionPoint, InsertionType, OperationPriority, \
    InsertionPointGraph, InsertionPointGraphNodeType
from tests.test_helpers import TwoConvTestModel, BasicConvTestModel, check_correct_nncf_modules_replacement


def test_disable_shape_matching():
    class MatMulModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy_param = torch.nn.Parameter(torch.ones([1]))

        def forward(self, inputs):
            half1, half2 = torch.chunk(inputs, 2, dim=2)
            return torch.bmm(half1, half2.transpose(1, 2))

    model = MatMulModel()

    input_shape_1 = (3, 32, 32)
    input_shape_2 = (4, 64, 64)

    qnet_no_shape = NNCFNetwork(deepcopy(model), input_infos=[ModelInputInfo(input_shape_1), ],
                                scopes_without_shape_matching=['MatMulModel'])  # type: NNCFNetwork
    _ = qnet_no_shape(torch.zeros(*input_shape_1))
    graph_1 = deepcopy(qnet_no_shape.get_graph())

    _ = qnet_no_shape(torch.zeros(*input_shape_2))
    graph_2 = deepcopy(qnet_no_shape.get_graph())

    keys_1 = list(graph_1.get_all_node_keys())
    keys_2 = list(graph_2.get_all_node_keys())
    assert len(keys_1) == 2  # 1 input node + 1 operation node
    assert keys_1 == keys_2


    qnet = NNCFNetwork(model, input_infos=[ModelInputInfo(input_shape_1), ])  # type: NNCFNetwork
    _ = qnet(torch.zeros(*input_shape_1))
    _ = qnet(torch.zeros(*input_shape_2))
    # The second forward run should have led to an increase in registered node counts
    # since disable_shape_matching was False and the network was run with a different
    # shape of input tensor
    assert qnet.get_graph().get_nodes_count() > graph_1.get_nodes_count()


def test_check_correct_modules_replacement():
    model = TwoConvTestModel()
    nncf_model = NNCFNetwork(TwoConvTestModel(), input_infos=[ModelInputInfo([1, 1, 4, 4])])  # type: NNCFNetwork

    _, nncf_modules = check_correct_nncf_modules_replacement(model, nncf_model)
    assert set(nncf_modules) == set(nncf_model.get_nncf_modules())


# pylint: disable=protected-access
def test_find_node_in_nx_graph_by_scope():
    model = TwoConvTestModel()
    nncf_model = NNCFNetwork(deepcopy(model), input_infos=[ModelInputInfo([1, 1, 4, 4])])  # type: NNCFNetwork
    nncf_graph = nncf_model.get_original_graph()

    # Valid scopes should be successfully found
    valid_nncf_modules = nncf_model.get_nncf_modules()
    nodes_list = list(nncf_graph._nx_graph.nodes)
    for module_scope, _ in valid_nncf_modules.items():
        graph_node = nncf_graph.find_node_in_nx_graph_by_scope(module_scope)
        assert graph_node is not None
        assert isinstance(graph_node, dict)
        assert graph_node['key'] in nodes_list

    fake_model = BasicConvTestModel()
    fake_nncf_model = NNCFNetwork(deepcopy(fake_model), input_infos=[ModelInputInfo([1, 1, 4, 4])])

    # Not valid scopes shouldn't be found
    fake_nncf_modules = fake_nncf_model.get_nncf_modules()
    for module_scope, _ in fake_nncf_modules.items():
        graph_node = nncf_graph.find_node_in_nx_graph_by_scope(module_scope)
        assert graph_node is None


class InsertionPointTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 1, 1)
        self.linear_wts = nn.Parameter(torch.FloatTensor(size=(100, 100)))
        self.conv2 = nn.Conv2d(1, 1, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, input_):
        x = self.conv1(input_)
        x = x.flatten()
        x = nn.functional.linear(x, self.linear_wts)
        x = x.reshape((1, 1, 10, 10))
        x = self.conv2(x)
        x = self.relu(x)
        return x


class TestInsertionCommands:
    @pytest.fixture()
    def setup(self):
        self.compressed_model = NNCFNetwork(InsertionPointTestModel(),
                                            [ModelInputInfo((1, 1, 10, 10))])  # type: NNCFNetwork

    conv1_module_scope = Scope.from_str('InsertionPointTestModel/NNCFConv2d[conv1]')
    conv1_module_context = InputAgnosticOperationExecutionContext('', conv1_module_scope, 0)
    point_for_conv1_weights = InsertionPoint(ia_op_exec_context=conv1_module_context,
                                             insertion_type=InsertionType.NNCF_MODULE_PRE_OP)
    point_for_conv1_inputs = InsertionPoint(ia_op_exec_context=conv1_module_context,
                                            insertion_type=InsertionType.NNCF_MODULE_PRE_OP)
    point_for_conv1_activations = InsertionPoint(ia_op_exec_context=conv1_module_context,
                                                 insertion_type=InsertionType.NNCF_MODULE_POST_OP)

    conv2_module_scope = Scope.from_str('InsertionPointTestModel/NNCFConv2d[conv2]')
    conv2_module_context = InputAgnosticOperationExecutionContext('', conv2_module_scope, 0)
    point_for_conv2_weights = InsertionPoint(ia_op_exec_context=conv2_module_context,
                                             insertion_type=InsertionType.NNCF_MODULE_PRE_OP)
    point_for_conv2_inputs = InsertionPoint(ia_op_exec_context=conv2_module_context,
                                            insertion_type=InsertionType.NNCF_MODULE_PRE_OP)
    point_for_conv2_activations = InsertionPoint(ia_op_exec_context=conv2_module_context,
                                                 insertion_type=InsertionType.NNCF_MODULE_POST_OP)

    linear_op_scope = Scope.from_str('InsertionPointTestModel/linear_0')
    linear_op_context = InputAgnosticOperationExecutionContext('linear',
                                                               linear_op_scope,
                                                               0)
    point_for_linear_weight_input = InsertionPoint(ia_op_exec_context=linear_op_context,
                                                   insertion_type=InsertionType.OPERATOR_PRE_HOOK)
    point_for_linear_activation = InsertionPoint(ia_op_exec_context=linear_op_context,
                                                 insertion_type=InsertionType.OPERATOR_POST_HOOK)

    relu_op_scope = Scope.from_str('InsertionPointTestModel/ReLU[relu]/relu')
    relu_op_context = InputAgnosticOperationExecutionContext('relu',
                                                             relu_op_scope,
                                                             0)
    point_for_relu_inputs = InsertionPoint(ia_op_exec_context=relu_op_context,
                                           insertion_type=InsertionType.OPERATOR_PRE_HOOK)
    point_for_relu_activations = InsertionPoint(ia_op_exec_context=relu_op_context,
                                                insertion_type=InsertionType.OPERATOR_POST_HOOK)

    available_points = [point_for_conv1_weights,
                        point_for_conv2_weights,
                        point_for_conv1_inputs,
                        point_for_conv2_inputs,
                        point_for_conv1_activations,
                        point_for_conv2_activations,
                        point_for_linear_activation,
                        point_for_linear_weight_input,
                        point_for_relu_activations,
                        point_for_relu_inputs]

    @pytest.mark.parametrize("insertion_point", available_points)
    def test_single_insertions(self, setup, insertion_point):
        if insertion_point.insertion_type in [InsertionType.OPERATOR_PRE_HOOK, InsertionType.OPERATOR_POST_HOOK]:
            hook = lambda x: x
        else:
            hook = BaseOp(lambda x: x)

        command = InsertionCommand(insertion_point, hook)
        self.compressed_model.register_insertion_command(command)
        self.compressed_model.commit_compression_changes()

        #pylint:disable=protected-access
        if insertion_point.insertion_type == InsertionType.OPERATOR_PRE_HOOK:
            ctx = self.compressed_model.get_tracing_context()
            assert ctx._pre_hooks[command.insertion_point.ia_op_exec_context][0] is hook
        if insertion_point.insertion_type == InsertionType.OPERATOR_POST_HOOK:
            ctx = self.compressed_model.get_tracing_context()
            assert ctx._post_hooks[command.insertion_point.ia_op_exec_context][0] is hook
        if insertion_point.insertion_type == InsertionType.NNCF_MODULE_PRE_OP:
            module = self.compressed_model.get_module_by_scope(
                command.insertion_point.ia_op_exec_context.scope_in_model)
            assert module.pre_ops["0"] is hook

        if insertion_point.insertion_type == InsertionType.NNCF_MODULE_POST_OP:
            module = self.compressed_model.get_module_by_scope(
                command.insertion_point.ia_op_exec_context.scope_in_model)
            assert module.post_ops["0"] is hook

    priority_types = ["same", "different"]
    insertion_types = InsertionType
    priority_test_cases = list(itertools.product(priority_types, insertion_types))

    @staticmethod
    def check_order(iterable1: List, iterable2: List, ordering: List):
        for idx, order in enumerate(ordering):
            assert iterable1[idx] is iterable2[order]

    # pylint:disable=undefined-variable
    @pytest.mark.parametrize("case", priority_test_cases, ids=[x[1].name + '-' + x[0] for x in priority_test_cases])
    def test_priority(self, case, setup):
        #pylint:disable=too-many-branches
        priority_type = case[0]
        insertion_type = case[1]
        if insertion_type in [InsertionType.NNCF_MODULE_PRE_OP, InsertionType.NNCF_MODULE_POST_OP]:
            hook1 = BaseOp(lambda x: x)
            hook2 = BaseOp(lambda x: 2 * x)
            hook3 = BaseOp(lambda x: 3 * x)
        else:
            hook1 = lambda x: x
            hook2 = lambda x: 2 * x
            hook3 = lambda x: 3 * x

        if insertion_type == InsertionType.NNCF_MODULE_PRE_OP:
            point = self.point_for_conv2_weights
        elif insertion_type == InsertionType.NNCF_MODULE_POST_OP:
            point = self.point_for_conv1_activations
        elif insertion_type == InsertionType.OPERATOR_PRE_HOOK:
            point = self.point_for_linear_weight_input
        elif insertion_type == InsertionType.OPERATOR_POST_HOOK:
            point = self.point_for_relu_activations

        if priority_type == "same":
            # Same-priority commands will be executed in registration order
            command1 = InsertionCommand(point, hook1, OperationPriority.DEFAULT_PRIORITY)
            command2 = InsertionCommand(point, hook2, OperationPriority.DEFAULT_PRIORITY)
            command3 = InsertionCommand(point, hook3, OperationPriority.DEFAULT_PRIORITY)
        else:
            # Prioritized commands will be executed in ascending priority order
            command1 = InsertionCommand(point, hook1, OperationPriority.SPARSIFICATION_PRIORITY)
            command2 = InsertionCommand(point, hook2, OperationPriority.QUANTIZATION_PRIORITY)
            command3 = InsertionCommand(point, hook3, OperationPriority.DEFAULT_PRIORITY)

        self.compressed_model.register_insertion_command(command1)
        self.compressed_model.register_insertion_command(command2)
        self.compressed_model.register_insertion_command(command3)
        self.compressed_model.commit_compression_changes()

        hook_list = [hook1, hook2, hook3]

        if priority_type == "same":
            order = [0, 1, 2]
        elif priority_type == "different":
            order = [2, 0, 1]

        #pylint:disable=protected-access
        if insertion_type == InsertionType.OPERATOR_PRE_HOOK:
            ctx = self.compressed_model.get_tracing_context()
            self.check_order(ctx._pre_hooks[point.ia_op_exec_context], hook_list, order)
        if insertion_type == InsertionType.OPERATOR_POST_HOOK:
            ctx = self.compressed_model.get_tracing_context()
            self.check_order(ctx._post_hooks[point.ia_op_exec_context], hook_list, order)

        if insertion_type == InsertionType.NNCF_MODULE_PRE_OP:
            module = self.compressed_model.get_module_by_scope(point.ia_op_exec_context.scope_in_model)
            # Works because Pytorch ModuleDict is ordered
            self.check_order(list(module.pre_ops.values()), hook_list, order)

        if insertion_type == InsertionType.NNCF_MODULE_POST_OP:
            module = self.compressed_model.get_module_by_scope(point.ia_op_exec_context.scope_in_model)
            # Works because Pytorch ModuleDict is ordered
            self.check_order(list(module.post_ops.values()), hook_list, order)


def get_two_branch_mock_model_graph() -> nx.DiGraph:
    mock_node_attrs = get_mock_nncf_node_attrs()
    mock_graph = nx.DiGraph()

    #   (A)
    #    |
    #   (B)
    #  /   \
    # (C)   (D)
    # |     |
    # (E)   |
    #  \   /
    #   (F)
    #    |
    #   (G)
    #    |
    #   (H)

    node_keys = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    for node_key in node_keys:
        mock_graph.add_node(node_key, **mock_node_attrs)

    mock_graph.add_edges_from([('A', 'B'), ('B', 'C'), ('B', 'D'), ('C', 'E'), ('E', 'F'),
                               ('D', 'F'), ('F', 'G'), ('G', 'H')])
    return mock_graph


MOCK_OPERATOR_NAME = "conv_transpose2d"


def get_mock_nncf_node_attrs():
    return {
        NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR: OperationExecutionContext(MOCK_OPERATOR_NAME,
                                                                       Scope(),
                                                                       0,
                                                                       [None])
    }


class TestInsertionPointGraph:
    def test_insertion_point_setup(self, tmp_path):
        # TODO: Change testing premises when module pre/post-op hooks and input/output nodes
        # are correctly handled
        mock_graph = get_two_branch_mock_model_graph()

        ip_graph = InsertionPointGraph(mock_graph)
        nx.drawing.nx_pydot.write_dot(ip_graph, str(tmp_path / "test_ip_graph.dot"))

        ref_node_len = 3 * len(mock_graph.nodes)  # 2 additional nodes per each operator node
        ref_edge_len = 3 * len(mock_graph.edges)

        assert len(ip_graph.nodes) == ref_node_len
        assert len(ip_graph.edges) == ref_edge_len

        for node_key, node in mock_graph.nodes.items():
            ip_graph_op_node = ip_graph.nodes[node_key]
            assert ip_graph_op_node[InsertionPointGraph.NODE_TYPE_NODE_ATTR] == InsertionPointGraphNodeType.OPERATOR
            preds = list(ip_graph.predecessors(node_key))
            succs = list(ip_graph.successors(node_key))
            assert len(preds) == 1
            assert len(succs) == 1
            pre_hook_ip_node_key = preds[0]
            post_hook_ip_node_key = succs[0]
            pre_hook_ip_node = ip_graph.nodes[preds[0]]
            post_hook_ip_node = ip_graph.nodes[succs[0]]
            pre_hook_ip_node_type = pre_hook_ip_node[InsertionPointGraph.NODE_TYPE_NODE_ATTR]
            post_hook_ip_node_type = post_hook_ip_node[InsertionPointGraph.NODE_TYPE_NODE_ATTR]
            assert pre_hook_ip_node_type == InsertionPointGraphNodeType.INSERTION_POINT
            assert post_hook_ip_node_type == InsertionPointGraphNodeType.INSERTION_POINT
            ref_associated_ip_node_keys_set = {pre_hook_ip_node_key, post_hook_ip_node_key}
            assert ref_associated_ip_node_keys_set == ip_graph_op_node[
                InsertionPointGraph.ASSOCIATED_IP_NODE_KEYS_NODE_ATTR]
            original_neighbours = mock_graph.neighbors(node_key)
            for neighbour in original_neighbours:
                # IP node insertion should not disrupt the graph superstructure
                ip_graph_paths = list(nx.all_simple_paths(ip_graph, node_key, neighbour))
                for path in ip_graph_paths:
                    path = path[1:-1]
                    for path_node_key in path:
                        node = ip_graph.nodes[path_node_key]
                        node_type = node[InsertionPointGraph.NODE_TYPE_NODE_ATTR]
                        assert node_type == InsertionPointGraphNodeType.INSERTION_POINT

        for node_key, node in ip_graph.nodes.items():
            preds = list(ip_graph.predecessors(node_key))
            succs = list(ip_graph.successors(node_key))
            assert len(preds) != 0 or len(succs) != 0

        for from_node_key, to_node_key in ip_graph.edges.keys():
            assert from_node_key in ip_graph.nodes
            assert to_node_key in ip_graph.nodes

    def test_insertion_point_data_in_ip_nodes(self):
        # TODO: extend for modules
        mock_graph = nx.DiGraph()
        ref_op_exec_context = OperationExecutionContext("baz",
                                                        Scope.from_str("Test/Scope[foo]/bar"),
                                                        0,
                                                        [None])
        node_attrs = {
            NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR: ref_op_exec_context
        }

        node_key = 0
        mock_graph.add_node(node_key, **node_attrs)

        ip_graph = InsertionPointGraph(mock_graph)

        for node_key in mock_graph.nodes.keys():
            preds = list(ip_graph.predecessors(node_key))
            succs = list(ip_graph.successors(node_key))
            pre_hook_ip_node = ip_graph.nodes[preds[0]]
            post_hook_ip_node = ip_graph.nodes[succs[0]]

            pre_hook_ip = pre_hook_ip_node[InsertionPointGraph.INSERTION_POINT_DATA_NODE_ATTR]
            post_hook_ip = post_hook_ip_node[InsertionPointGraph.INSERTION_POINT_DATA_NODE_ATTR]
            assert pre_hook_ip.insertion_type == InsertionType.OPERATOR_PRE_HOOK
            assert post_hook_ip.insertion_type == InsertionType.OPERATOR_POST_HOOK

            assert pre_hook_ip.ia_op_exec_context == ref_op_exec_context.input_agnostic
            assert post_hook_ip.ia_op_exec_context == ref_op_exec_context.input_agnostic

    def test_operator_metatype_marking(self):
        from nncf.dynamic_graph.operator_metatypes import Conv2dMetatype, BatchNormMetatype, RELUMetatype, \
            MaxPool2dMetatype, \
            ConvTranspose2dMetatype, DepthwiseConv2dSubtype, AddMetatype, AvgPool2dMetatype, LinearMetatype
        ref_scope_vs_metatype_dict = {
            "/" + MODEL_INPUT_OP_NAME + "_0": NoopMetatype,
            "ModelForMetatypeTesting/NNCFConv2d[conv_regular]/conv2d_0": Conv2dMetatype,
            "ModelForMetatypeTesting/BatchNorm2d[bn]/batch_norm_0": BatchNormMetatype,
            "ModelForMetatypeTesting/RELU_0": RELUMetatype,
            "ModelForMetatypeTesting/MaxPool2d[max_pool2d]/max_pool2d_0": MaxPool2dMetatype,
            "ModelForMetatypeTesting/NNCFConvTranspose2d[conv_transpose]/conv_transpose2d_0": ConvTranspose2dMetatype,
            "ModelForMetatypeTesting/NNCFConv2d[conv_depthwise]/conv2d_0": DepthwiseConv2dSubtype,
            "ModelForMetatypeTesting/__iadd___0": AddMetatype,
            "ModelForMetatypeTesting/AdaptiveAvgPool2d[adaptive_avg_pool]/adaptive_avg_pool2d_0": AvgPool2dMetatype,
            "ModelForMetatypeTesting/NNCFLinear[linear]/linear_0": LinearMetatype
        }
        class ModelForMetatypeTesting(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_regular = torch.nn.Conv2d(in_channels=3,
                                                    out_channels=16,
                                                    kernel_size=3)
                self.bn = torch.nn.BatchNorm2d(num_features=16)
                self.max_pool2d = torch.nn.MaxPool2d(kernel_size=2)
                self.conv_transpose = torch.nn.ConvTranspose2d(in_channels=16,
                                                               out_channels=8,
                                                               kernel_size=3)
                self.conv_depthwise = torch.nn.Conv2d(in_channels=8, out_channels=8,
                                                      kernel_size=5, groups=8)
                self.adaptive_avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=1)
                self.linear = torch.nn.Linear(in_features=8, out_features=1)

            def forward(self, input_):
                x = self.conv_regular(input_)
                x = self.bn(x)
                x = torch.nn.functional.relu(x)
                x.transpose_(2, 3)
                x = self.max_pool2d(x)
                x = self.conv_transpose(x)
                x = self.conv_depthwise(x)
                x += torch.ones_like(x)
                x = self.adaptive_avg_pool(x)
                x = self.linear(x.flatten())
                return x

        model = ModelForMetatypeTesting()
        nncf_network = NNCFNetwork(model, [ModelInputInfo((1, 3, 300, 300))])
        ip_graph = nncf_network.get_insertion_point_graph()

        for node in ip_graph.nodes().values():
            if node[InsertionPointGraph.NODE_TYPE_NODE_ATTR] == InsertionPointGraphNodeType.OPERATOR:
                nncf_node_ref = node[InsertionPointGraph.REGULAR_NODE_REF_NODE_ATTR]
                scope_str = str(nncf_node_ref[NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR].input_agnostic)
                assert scope_str in ref_scope_vs_metatype_dict
                ref_metatype = ref_scope_vs_metatype_dict[scope_str]
                assert node[InsertionPointGraph.OPERATOR_METATYPE_NODE_ATTR] == ref_metatype
