# pylint:disable=too-many-lines
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
import random
from collections import namedtuple
from typing import Dict, List, Tuple

import networkx as nx
import pytest

from nncf.dynamic_graph.context import Scope
from nncf.dynamic_graph.graph import OperationExecutionContext, NNCFGraph
from nncf.dynamic_graph.version_agnostic_op_names import get_version_agnostic_name
from nncf.nncf_network import InsertionPointGraph, InsertionInfo, InsertionPointGraphNodeType
from nncf.quantization.layers import QuantizerConfig, QuantizationMode
from nncf.quantization.quantizer_propagation import QuantizerPropagationStateGraph as QPSG, \
    QuantizerPropagationStateGraphNodeType, QuantizationTrait, OPERATOR_METATYPES, DEFAULT_QUANT_TRAIT_TO_OP_DICT, \
    QuantizerPropagationSolver, TransitionStatus, PropagationStrategy, PropagatingQuantizer
from tests.quantization.test_quantizer_propagation_graph import get_edge_paths_for_propagation
from tests.test_nncf_network import get_mock_nncf_node_attrs


class TestQuantizerPropagationSolver:
    @staticmethod
    def get_mock_model_node_attrs_for_op_name(op_name: str) -> OperationExecutionContext:
        return OperationExecutionContext(op_name,
                                         Scope(),
                                         0,
                                         [None])

    @staticmethod
    def get_randomly_connected_model_graph(op_name_keys: List[str]) -> nx.DiGraph:
        graph_len = len(op_name_keys)
        mock_graph = nx.generators.gnc_graph(graph_len, seed=0)
        shuffled_op_names = random.sample(op_name_keys, len(op_name_keys))
        for idx, (_, node) in enumerate(mock_graph.nodes.items()):
            node[NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR] = \
                TestQuantizerPropagationSolver.get_mock_model_node_attrs_for_op_name(shuffled_op_names[idx])
        return mock_graph

    @staticmethod
    def get_sequentially_connected_model_graph(op_name_keys: List[str]) -> nx.DiGraph:
        graph = nx.DiGraph()
        for node_key in op_name_keys:
            attrs = {
                NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR:
                    TestQuantizerPropagationSolver.get_mock_model_node_attrs_for_op_name(node_key)
            }
            graph.add_node(node_key, **attrs)

        edges = [(op_name_keys[i], op_name_keys[i + 1]) for i in range(0, len(op_name_keys) - 1)]
        for from_key, to_key in edges:
            graph.add_edge(from_key, to_key)
        return graph

    def test_quantization_traits_are_unambiguous_for_op_names(self):
        op_name_to_trait_dict = {}  # type: Dict[str, QuantizationTrait]
        for trait, arches in DEFAULT_QUANT_TRAIT_TO_OP_DICT.items():
            for op_meta in arches:
                aliases = op_meta.get_all_aliases()
                for alias in aliases:
                    if alias in op_name_to_trait_dict:
                        assert op_name_to_trait_dict[alias] == trait
                    else:
                        op_name_to_trait_dict[alias] = trait

    def test_set_quantization_traits_for_quant_prop_graph_nodes(self):
        # Test all patchable metatypes. If a patchable metatype is not registered
        # in quantization trait-to-metatype dict, the test will fail.
        tested_op_metatypes = list(OPERATOR_METATYPES.registry_dict.values()) # type: List[OperatorMetatype]
        tested_op_names = []
        for op_meta in tested_op_metatypes:
            aliases = op_meta.get_all_aliases()
            for alias in aliases:
                tested_op_names.append(get_version_agnostic_name(alias))

        # Edges should be irrelevant - using random graph
        mock_graph = self.get_randomly_connected_model_graph(tested_op_names)
        ip_graph = InsertionPointGraph(mock_graph)
        for node in ip_graph.nodes.values():
            if node[InsertionPointGraph.NODE_TYPE_NODE_ATTR] == InsertionPointGraphNodeType.OPERATOR:
                op_exec_context = node[InsertionPointGraph.REGULAR_NODE_REF_NODE_ATTR][
                    NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR]
                op_name = op_exec_context.operator_name
                ref_meta = OPERATOR_METATYPES.get_operator_metatype_by_op_name(op_name)
                node[InsertionPointGraph.OPERATOR_METATYPE_NODE_ATTR] = ref_meta

        quant_prop_graph = QPSG(ip_graph)
        quant_prop_solver = QuantizerPropagationSolver()
        quant_prop_graph = quant_prop_solver.set_allowed_quantization_types_for_operator_nodes(quant_prop_graph)
        op_quant_traits_map = quant_prop_solver.get_operator_quantization_traits_map()

        for qpg_node in quant_prop_graph.nodes().values():
            if qpg_node[QPSG.NODE_TYPE_NODE_ATTR] == QuantizerPropagationStateGraphNodeType.OPERATOR:
                quant_det_id = qpg_node[QPSG.OPERATOR_METATYPE_NODE_ATTR]
                quant_types = qpg_node[QPSG.ALLOWED_INPUT_QUANTIZATION_TYPES_NODE_ATTR]
                if op_quant_traits_map[quant_det_id] == QuantizationTrait.INPUTS_QUANTIZABLE:
                    # TODO: check for correspondence of operator type and HW config to initial
                    # quantization types
                    assert quant_types == QuantizerPropagationSolver.DEFAULT_QUANTIZATION_TYPES

    def test_setup_initial_quantizers_in_quant_prop_graph(self):
        ops_to_quantize = ['conv2d', 'matmul', 'gelu']
        ops_not_to_quantize = ['batch_norm', 'max_pool2d', 'dropout', 'min', 'softmax']
        node_keys = ops_to_quantize + ops_not_to_quantize
        mock_graph = self.get_sequentially_connected_model_graph(node_keys)

        ip_graph = InsertionPointGraph(mock_graph)
        for node in ip_graph.nodes.values():
            if node[InsertionPointGraph.NODE_TYPE_NODE_ATTR] == InsertionPointGraphNodeType.OPERATOR:
                op_exec_context = node[InsertionPointGraph.REGULAR_NODE_REF_NODE_ATTR][
                    NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR]
                op_name = op_exec_context.operator_name
                ref_meta = OPERATOR_METATYPES.get_operator_metatype_by_op_name(op_name)
                node[InsertionPointGraph.OPERATOR_METATYPE_NODE_ATTR] = ref_meta

        qp_graph = QPSG(ip_graph)
        quant_prop_solver = QuantizerPropagationSolver()
        qp_graph = quant_prop_solver.set_allowed_quantization_types_for_operator_nodes(qp_graph)
        qp_graph = quant_prop_solver.setup_initial_quantizers(qp_graph)

        for node_key in ops_to_quantize:
            pred_ip_key = next(qp_graph.predecessors(node_key))
            node = qp_graph.nodes[node_key]
            pred_ip_node = qp_graph.nodes[pred_ip_key]
            prop_quant = pred_ip_node[QPSG.PROPAGATING_QUANTIZER_NODE_ATTR]
            assert prop_quant is not None
            assert node[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR] == [prop_quant]

            edge = qp_graph.edges[pred_ip_key, node_key]
            assert edge[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR] == [prop_quant]

        for node_key in ops_not_to_quantize:
            pred_ip_key = next(qp_graph.predecessors(node_key))
            node = qp_graph.nodes[node_key]
            pred_ip_node = qp_graph.nodes[pred_ip_key]
            assert pred_ip_node[QPSG.PROPAGATING_QUANTIZER_NODE_ATTR] is None

            assert not node[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
            edge = qp_graph.edges[pred_ip_key, node_key]
            assert not edge[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]

    MergeQConfigTestStruct = namedtuple('MergeQConfigTestStruct',
                                        ('master_config_list_before_merge',
                                         'slave_config_list_dict_before_merge',
                                         'master_config_list_after_merge',
                                         'slave_config_list_dict_after_merge'))
    QCONFIG_MASTER_SLAVE_BEFORE_AND_AFTER_MERGING = [
        # Compatible configs on all branches
        MergeQConfigTestStruct(
            master_config_list_before_merge=[QuantizerConfig(bits=8)],
            slave_config_list_dict_before_merge={
                "foo": [QuantizerConfig(bits=8)],
                "bar": [QuantizerConfig(bits=8)]
            },
            master_config_list_after_merge=[QuantizerConfig(bits=8)],
            slave_config_list_dict_after_merge={
                "foo": [QuantizerConfig(bits=8)],
                "bar": [QuantizerConfig(bits=8)]
            }),
        MergeQConfigTestStruct(
            master_config_list_before_merge=[QuantizerConfig(bits=6,
                                                             mode=QuantizationMode.ASYMMETRIC)],
            slave_config_list_dict_before_merge={
                "foo": [QuantizerConfig(bits=6,
                                        mode=QuantizationMode.ASYMMETRIC)],
                "bar": [QuantizerConfig(bits=6,
                                        mode=QuantizationMode.ASYMMETRIC)],
                "baz": [QuantizerConfig(bits=6,
                                        mode=QuantizationMode.ASYMMETRIC)]
            },
            master_config_list_after_merge=[QuantizerConfig(bits=6,
                                                            mode=QuantizationMode.ASYMMETRIC)],
            slave_config_list_dict_after_merge={
                "foo": [QuantizerConfig(bits=6,
                                        mode=QuantizationMode.ASYMMETRIC)],
                "bar": [QuantizerConfig(bits=6,
                                        mode=QuantizationMode.ASYMMETRIC)],
                "baz": [QuantizerConfig(bits=6,
                                        mode=QuantizationMode.ASYMMETRIC)]
            }),

        # Precision narrowed relative to master config on some branches, but master
        # config is still compatible
        MergeQConfigTestStruct(
            master_config_list_before_merge=[QuantizerConfig(bits=6)],
            slave_config_list_dict_before_merge={
                "foo": [QuantizerConfig(bits=4)],
                "bar": [QuantizerConfig(bits=5)]
            },
            master_config_list_after_merge=[QuantizerConfig(bits=6)],
            slave_config_list_dict_after_merge={
                "foo": [QuantizerConfig(bits=4)],
                "bar": [QuantizerConfig(bits=5)]
            }),

        MergeQConfigTestStruct(
            master_config_list_before_merge=[QuantizerConfig(bits=8,
                                                             mode=QuantizationMode.ASYMMETRIC)],
            slave_config_list_dict_before_merge={
                "foo": [QuantizerConfig(bits=4)],
                "bar": [QuantizerConfig(bits=5)]
            },
            master_config_list_after_merge=[QuantizerConfig(bits=8,
                                                            mode=QuantizationMode.ASYMMETRIC)],
            slave_config_list_dict_after_merge={
                "foo": [QuantizerConfig(bits=4)],
                "bar": [QuantizerConfig(bits=5)]
            }),

        # Potential master configs excluded due to conflict with a branch
        MergeQConfigTestStruct(
            master_config_list_before_merge=[QuantizerConfig(bits=8),
                                             QuantizerConfig(bits=6)],
            slave_config_list_dict_before_merge={
                "foo": [QuantizerConfig(bits=7)],
                "bar": [QuantizerConfig(bits=8)],
                "baz": [QuantizerConfig(bits=7)]
            },
            master_config_list_after_merge=[QuantizerConfig(bits=8)],
            slave_config_list_dict_after_merge={
                "foo": [QuantizerConfig(bits=7)],
                "bar": [QuantizerConfig(bits=8)],
                "baz": [QuantizerConfig(bits=7)]
            }),

        MergeQConfigTestStruct(
            master_config_list_before_merge=[QuantizerConfig(bits=7),
                                             QuantizerConfig(bits=7,
                                                             mode=QuantizationMode.ASYMMETRIC)],
            slave_config_list_dict_before_merge={
                "foo": [QuantizerConfig(bits=7,
                                        mode=QuantizationMode.ASYMMETRIC)],
                "bar": [QuantizerConfig(bits=7,
                                        mode=QuantizationMode.ASYMMETRIC)]
            },
            master_config_list_after_merge=[QuantizerConfig(bits=7,
                                                            mode=QuantizationMode.ASYMMETRIC)],
            slave_config_list_dict_after_merge={
                "foo": [QuantizerConfig(bits=7,
                                        mode=QuantizationMode.ASYMMETRIC)],
                "bar": [QuantizerConfig(bits=7,
                                        mode=QuantizationMode.ASYMMETRIC)]
            }),
        MergeQConfigTestStruct(
            master_config_list_before_merge=[QuantizerConfig(bits=7),
                                             QuantizerConfig(bits=7,
                                                             mode=QuantizationMode.ASYMMETRIC)],
            slave_config_list_dict_before_merge={
                "foo": [QuantizerConfig(bits=7,
                                        mode=QuantizationMode.ASYMMETRIC)],
                "bar": [QuantizerConfig(bits=7,
                                        mode=QuantizationMode.ASYMMETRIC)]
            },
            master_config_list_after_merge=[QuantizerConfig(bits=7,
                                                            mode=QuantizationMode.ASYMMETRIC)],
            slave_config_list_dict_after_merge={
                "foo": [QuantizerConfig(bits=7,
                                        mode=QuantizationMode.ASYMMETRIC)],
                "bar": [QuantizerConfig(bits=7,
                                        mode=QuantizationMode.ASYMMETRIC)]
            }),

        # Master config propagation-induced config exclusion on branches:
        MergeQConfigTestStruct(
            master_config_list_before_merge=[QuantizerConfig(bits=6)],
            slave_config_list_dict_before_merge={
                "foo": [QuantizerConfig(bits=8),
                        QuantizerConfig(bits=7),
                        QuantizerConfig(bits=6),],
                "bar": [QuantizerConfig(bits=8),
                        QuantizerConfig(bits=5)],
            },
            master_config_list_after_merge=[QuantizerConfig(bits=6)],
            slave_config_list_dict_after_merge={
                "foo": [QuantizerConfig(bits=6)],
                "bar": [QuantizerConfig(bits=5)]
            }),

        MergeQConfigTestStruct(
            master_config_list_before_merge=[QuantizerConfig(bits=6)],
            slave_config_list_dict_before_merge={
                "foo": [QuantizerConfig(bits=7),
                        QuantizerConfig(bits=6),
                        QuantizerConfig(bits=6,
                                        mode=QuantizationMode.ASYMMETRIC)],
                "bar": [QuantizerConfig(bits=8),
                        QuantizerConfig(bits=5),
                        QuantizerConfig(bits=4,
                                        mode=QuantizationMode.ASYMMETRIC)
                        ],
            },
            master_config_list_after_merge=[QuantizerConfig(bits=6)],
            slave_config_list_dict_after_merge={
                "foo": [QuantizerConfig(bits=6)],
                "bar": [QuantizerConfig(bits=5)]
            }),

        # Cases with conflicts resulting in no master configs left after merge and,
        # consequently, no propagation:
        MergeQConfigTestStruct(
            master_config_list_before_merge=[QuantizerConfig(bits=3)],
            slave_config_list_dict_before_merge={
                "foo": [QuantizerConfig(bits=7),
                        QuantizerConfig(bits=6),
                        QuantizerConfig(bits=6,
                                        mode=QuantizationMode.ASYMMETRIC)],
                "bar": [QuantizerConfig(bits=8),
                        QuantizerConfig(bits=5),
                        QuantizerConfig(bits=4,
                                        mode=QuantizationMode.ASYMMETRIC)
                        ],
            },
            master_config_list_after_merge=[],
            slave_config_list_dict_after_merge={}),

        MergeQConfigTestStruct(
            master_config_list_before_merge=[QuantizerConfig(bits=8),
                                             QuantizerConfig(bits=4,
                                                             mode=QuantizationMode.ASYMMETRIC)],
            slave_config_list_dict_before_merge={
                "foo": [QuantizerConfig(bits=7,
                                        mode=QuantizationMode.ASYMMETRIC),
                        QuantizerConfig(bits=6,
                                        mode=QuantizationMode.ASYMMETRIC)],
                "bar": [QuantizerConfig(bits=8),
                        QuantizerConfig(bits=5)
                        ],
            },
            master_config_list_after_merge=[],
            slave_config_list_dict_after_merge={})

        # TODO: extend with signed/unsigned test cases
    ]

    @staticmethod
    @pytest.fixture(params=QCONFIG_MASTER_SLAVE_BEFORE_AND_AFTER_MERGING)
    def qconfig_merge_test_struct(request):
        return request.param

    def test_get_merged_qconfigs(self, qconfig_merge_test_struct):
        quant_prop_solver = QuantizerPropagationSolver()
        ref_merged_master_config_list = qconfig_merge_test_struct.master_config_list_after_merge
        ref_merged_slave_config_dict_list = qconfig_merge_test_struct.slave_config_list_dict_after_merge

        merged_master_config_list, merged_slave_config_dict_list = quant_prop_solver.get_merged_qconfigs(
            qconfig_merge_test_struct.master_config_list_before_merge,
            qconfig_merge_test_struct.slave_config_list_dict_before_merge
        )

        assert ref_merged_master_config_list == merged_master_config_list
        assert ref_merged_slave_config_dict_list == merged_slave_config_dict_list


    def get_branching_model_graph(self):
        mock_node_attrs = get_mock_nncf_node_attrs()
        mock_graph = nx.DiGraph()

        #     (A)
        #      |
        #     (B)
        #   /  |  \
        # (C) (D) (E)
        #  |       | \
        # (F)     (G) (H)
        #           \ /
        #           (I)
        #            |
        #           (J)

        node_keys = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        for node_key in node_keys:
            mock_graph.add_node(node_key, **mock_node_attrs)

        mock_graph.add_edges_from([('A', 'B'), ('B', 'C'), ('B', 'D'), ('B', 'E'), ('C', 'F'),
                                   ('E', 'G'), ('E', 'H'), ('G', 'I'), ('H', 'I'), ('I', 'J')])
        return mock_graph

    BranchTransitionTestStruct = namedtuple('BranchTransitionTestStruct',
                                            (  # Unspecified nodes are marked as quantization agnostic
                                                'init_node_to_trait_and_configs_dict',
                                                'starting_master_quantizer_ip_node',
                                                'target_branching_node_for_master_quantizer',
                                                'strategy_vs_expected_status_dict'))

    BRANCH_TRANSITION_TEST_CASES = [
        # Downward branches are quantization-agnostic
        BranchTransitionTestStruct(
            init_node_to_trait_and_configs_dict=
            {
                'D': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()]),
            },
            starting_master_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('D'),
            target_branching_node_for_master_quantizer=InsertionPointGraph.get_post_hook_node_key('B'),
            strategy_vs_expected_status_dict={
                PropagationStrategy.CONSERVATIVE: None,
                PropagationStrategy.AGGRESSIVE: None,
            }
        ),

        # Downward branches have compatible quantization configs
        BranchTransitionTestStruct(
            init_node_to_trait_and_configs_dict=
            {
                'F': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()]),
                'D': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()]),
                'E': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()]),
            },
            starting_master_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('F'),
            target_branching_node_for_master_quantizer=InsertionPointGraph.get_post_hook_node_key('B'),
            strategy_vs_expected_status_dict={
                PropagationStrategy.CONSERVATIVE: None,
                PropagationStrategy.AGGRESSIVE: None,
            }
        ),

        # A branch has a non-quantizable op
        BranchTransitionTestStruct(
            init_node_to_trait_and_configs_dict=
            {
                'F': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()]),
                'D': (QuantizationTrait.NON_QUANTIZABLE,
                      []),
                'E': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()]),
            },
            starting_master_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('E'),
            target_branching_node_for_master_quantizer=InsertionPointGraph.get_post_hook_node_key('B'),
            strategy_vs_expected_status_dict={
                PropagationStrategy.CONSERVATIVE: TransitionStatus.SHOULD_NOT_TRANSITION,
                PropagationStrategy.AGGRESSIVE: TransitionStatus.SHOULD_NOT_TRANSITION,
            }
        ),

        BranchTransitionTestStruct(
            init_node_to_trait_and_configs_dict=
            {
                'F': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()]),
                'D': (QuantizationTrait.NON_QUANTIZABLE,
                      []),
                'J': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()]),
            },
            starting_master_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('F'),
            target_branching_node_for_master_quantizer=InsertionPointGraph.get_post_hook_node_key('B'),
            strategy_vs_expected_status_dict={
                PropagationStrategy.CONSERVATIVE: TransitionStatus.SHOULD_NOT_TRANSITION,
                PropagationStrategy.AGGRESSIVE: TransitionStatus.SHOULD_NOT_TRANSITION,
            }
        ),


        BranchTransitionTestStruct(
            init_node_to_trait_and_configs_dict=
            {
                'F': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()]),
                'D': (QuantizationTrait.NON_QUANTIZABLE,
                      []),
                'E': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()]),
            },
            starting_master_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('E'),
            target_branching_node_for_master_quantizer=InsertionPointGraph.get_post_hook_node_key('B'),
            strategy_vs_expected_status_dict={
                PropagationStrategy.CONSERVATIVE: TransitionStatus.SHOULD_NOT_TRANSITION,
                PropagationStrategy.AGGRESSIVE: TransitionStatus.SHOULD_NOT_TRANSITION,
            }
        ),

        BranchTransitionTestStruct(
            init_node_to_trait_and_configs_dict=
            {
                'F': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()]),
                'D': (QuantizationTrait.NON_QUANTIZABLE,
                      []),
                'E': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()]),
            },
            starting_master_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('E'),
            target_branching_node_for_master_quantizer=InsertionPointGraph.get_post_hook_node_key('B'),
            strategy_vs_expected_status_dict={
                PropagationStrategy.CONSERVATIVE: TransitionStatus.SHOULD_NOT_TRANSITION,
                PropagationStrategy.AGGRESSIVE: TransitionStatus.SHOULD_NOT_TRANSITION,
            }
        ),

        # All master configs are incompatible with branch configs
        BranchTransitionTestStruct(
            init_node_to_trait_and_configs_dict=
            {
                'F': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=7)]),
                'D': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=8), QuantizerConfig(bits=6)]),
                'E': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=4)]),
            },
            starting_master_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('E'),
            target_branching_node_for_master_quantizer=InsertionPointGraph.get_post_hook_node_key('B'),
            strategy_vs_expected_status_dict={
                PropagationStrategy.CONSERVATIVE: TransitionStatus.SHOULD_NOT_TRANSITION,
                PropagationStrategy.AGGRESSIVE: TransitionStatus.SHOULD_NOT_TRANSITION,
            }
        ),

        BranchTransitionTestStruct(
            init_node_to_trait_and_configs_dict=
            {
                'F': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=7, mode=QuantizationMode.ASYMMETRIC)]),
                'D': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=8), QuantizerConfig(bits=6, mode=QuantizationMode.ASYMMETRIC)]),
                'E': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=4), QuantizerConfig(bits=6)]),
            },
            starting_master_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('E'),
            target_branching_node_for_master_quantizer=InsertionPointGraph.get_post_hook_node_key('B'),
            strategy_vs_expected_status_dict={
                PropagationStrategy.CONSERVATIVE: TransitionStatus.SHOULD_NOT_TRANSITION,
                PropagationStrategy.AGGRESSIVE: TransitionStatus.SHOULD_NOT_TRANSITION,
            }
        ),

        # Compatible quantizers exist on the branches, but each is below an incompatible quantizer
        BranchTransitionTestStruct(
            init_node_to_trait_and_configs_dict=
            {
                'D': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=4), QuantizerConfig(bits=6)]),

                'C': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=7, mode=QuantizationMode.ASYMMETRIC)]),

                'E': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=8), QuantizerConfig(bits=6, mode=QuantizationMode.ASYMMETRIC)]),

                'F': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=4)]),
                'G': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=4)]),
                'H': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=4)]),

            },
            starting_master_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('D'),
            target_branching_node_for_master_quantizer=InsertionPointGraph.get_post_hook_node_key('B'),
            strategy_vs_expected_status_dict={
                PropagationStrategy.CONSERVATIVE: TransitionStatus.SHOULD_NOT_TRANSITION,
                PropagationStrategy.AGGRESSIVE: TransitionStatus.SHOULD_NOT_TRANSITION,
            }
        ),

        # Master config options narrowing due to transition, but otherwise transition is permitted
        BranchTransitionTestStruct(
            init_node_to_trait_and_configs_dict=
            {
                'F': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=5)]),
                'D': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=6)]),
                'E': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=4), QuantizerConfig(bits=6)]),
            },
            starting_master_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('E'),
            target_branching_node_for_master_quantizer=InsertionPointGraph.get_post_hook_node_key('B'),
            strategy_vs_expected_status_dict={
                PropagationStrategy.CONSERVATIVE: None,
                PropagationStrategy.AGGRESSIVE: None
            }
        ),

        # Branch config options narrowing due to transition - do not transition if the strategy
        # is conservative
        BranchTransitionTestStruct(
            init_node_to_trait_and_configs_dict=
            {
                'F': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=7, mode=QuantizationMode.ASYMMETRIC)]),
                'D': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=6), QuantizerConfig(bits=8)]),
                'E': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=4), QuantizerConfig(bits=6)]),
            },
            starting_master_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('F'),
            target_branching_node_for_master_quantizer=InsertionPointGraph.get_post_hook_node_key('B'),
            strategy_vs_expected_status_dict={
                PropagationStrategy.CONSERVATIVE: TransitionStatus.SHOULD_NOT_TRANSITION,
                PropagationStrategy.AGGRESSIVE: None
            }
        ),
    ]

    @staticmethod
    @pytest.fixture(params=BRANCH_TRANSITION_TEST_CASES)
    def branch_transition_test_struct(request):
        return request.param

    def test_check_branching_transition(self, branch_transition_test_struct: BranchTransitionTestStruct):
        init_node_to_trait_and_configs_dict = branch_transition_test_struct.init_node_to_trait_and_configs_dict
        starting_master_quantizer_ip_node = branch_transition_test_struct.starting_master_quantizer_ip_node
        target_node = branch_transition_test_struct.target_branching_node_for_master_quantizer
        strategy_vs_status = branch_transition_test_struct.strategy_vs_expected_status_dict

        # Graph preparation
        mock_graph = self.get_branching_model_graph()
        ip_graph = InsertionPointGraph(mock_graph)
        quant_prop_graph = QPSG(ip_graph)
        for node in quant_prop_graph.nodes.values():
            node[QPSG.QUANTIZATION_TRAIT_NODE_ATTR] = QuantizationTrait.QUANTIZATION_AGNOSTIC

        master_prop_quant = None
        for node_key, trait_and_configs_tuple in init_node_to_trait_and_configs_dict.items():
            trait = trait_and_configs_tuple[0]
            qconfigs = trait_and_configs_tuple[1]
            quant_prop_graph.nodes[node_key][QPSG.QUANTIZATION_TRAIT_NODE_ATTR] = trait
            if trait == QuantizationTrait.INPUTS_QUANTIZABLE:
                ip_node_key = InsertionPointGraph.get_pre_hook_node_key(node_key)
                prop_quant = quant_prop_graph.add_propagating_quantizer(qconfigs,
                                                                        ip_node_key)
                if ip_node_key == starting_master_quantizer_ip_node:
                    master_prop_quant = prop_quant

        path = get_edge_paths_for_propagation(quant_prop_graph,
                                              target_node,
                                              starting_master_quantizer_ip_node)
        master_prop_quant = quant_prop_graph.propagate_quantizer_via_path(master_prop_quant,
                                                                          path[0])

        # The propagating quantizers are in place, now check the transition
        for strategy, ref_status in strategy_vs_status.items():
            solver = QuantizerPropagationSolver(propagation_strategy=strategy)
            status = solver.check_branching_transition(quant_prop_graph,
                                                       master_prop_quant,
                                                       target_node)
            assert status == ref_status

    PathTransitionTestStruct = namedtuple('PathTransitionTestStruct',
                                          ('init_node_to_trait_configs_and_target_node_dict',
                                           # Unspecified nodes are marked as quantization agnostic
                                           'starting_master_quantizer_ip_node',
                                           'master_quantizer_qconfigs',
                                           'target_node_for_master_quantizer',
                                           'strategy_vs_expected_status_dict'))

    @staticmethod
    def prepare_propagation_graph_state(ip_graph: InsertionPointGraph,
                                        init_node_to_trait_configs_and_target_node_dict: Dict[
                                            str, Tuple]) -> Tuple[List[PropagatingQuantizer], QPSG]:
        quant_prop_graph = QPSG(ip_graph)
        prop_quantizers = []
        for node in quant_prop_graph.nodes.values():
            node[QPSG.QUANTIZATION_TRAIT_NODE_ATTR] = QuantizationTrait.QUANTIZATION_AGNOSTIC

        for node_key, trait_configs_and_target_tuple in init_node_to_trait_configs_and_target_node_dict.items():
            trait = trait_configs_and_target_tuple[0]
            qconfigs = trait_configs_and_target_tuple[1]
            target_node = trait_configs_and_target_tuple[2]
            quant_prop_graph.nodes[node_key][QPSG.QUANTIZATION_TRAIT_NODE_ATTR] = trait
            if trait == QuantizationTrait.INPUTS_QUANTIZABLE:
                ip_node_key = InsertionPointGraph.get_pre_hook_node_key(node_key)
                prop_quant = quant_prop_graph.add_propagating_quantizer(qconfigs,
                                                                        ip_node_key)
                path = get_edge_paths_for_propagation(quant_prop_graph,
                                                      target_node,
                                                      ip_node_key)
                prop_quant = quant_prop_graph.propagate_quantizer_via_path(prop_quant, path[0])
                prop_quantizers.append(prop_quant)

        return prop_quantizers, quant_prop_graph

    PATH_TRANSITION_TEST_CASES = [
        # Transition cases

        # Single propagating quantizer
        PathTransitionTestStruct(
            init_node_to_trait_configs_and_target_node_dict=
            {
            },
            starting_master_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('J'),
            master_quantizer_qconfigs=[QuantizerConfig()],
            target_node_for_master_quantizer=InsertionPointGraph.get_post_hook_node_key('E'),
            strategy_vs_expected_status_dict={
                PropagationStrategy.CONSERVATIVE: TransitionStatus.SHOULD_TRANSITION,
                PropagationStrategy.AGGRESSIVE: TransitionStatus.SHOULD_TRANSITION,
            }
        ),

        # Non-intersecting paths, no branch influence
        PathTransitionTestStruct(
            init_node_to_trait_configs_and_target_node_dict=
            {
                'D': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_post_hook_node_key('A')),
            },
            starting_master_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('F'),
            master_quantizer_qconfigs=[QuantizerConfig()],
            target_node_for_master_quantizer=InsertionPointGraph.get_post_hook_node_key('C'),
            strategy_vs_expected_status_dict={
                PropagationStrategy.CONSERVATIVE: TransitionStatus.SHOULD_TRANSITION,
                PropagationStrategy.AGGRESSIVE: TransitionStatus.SHOULD_TRANSITION,
            }
        ),

        # Non-intersecting paths, branch influence
        PathTransitionTestStruct(
            init_node_to_trait_configs_and_target_node_dict=
            {
                'F': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=6)],
                      InsertionPointGraph.get_pre_hook_node_key('C')),
                'G': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_pre_hook_node_key('E')),
                'H': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_pre_hook_node_key('E')),
            },
            starting_master_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('D'),
            master_quantizer_qconfigs=[QuantizerConfig()],
            target_node_for_master_quantizer=InsertionPointGraph.get_post_hook_node_key('A'),
            strategy_vs_expected_status_dict={
                PropagationStrategy.CONSERVATIVE: TransitionStatus.SHOULD_TRANSITION,
                PropagationStrategy.AGGRESSIVE: TransitionStatus.SHOULD_TRANSITION,
            }
        ),

        # Non-intersecting paths, branch influence with downward branch config narrowing
        PathTransitionTestStruct(
            init_node_to_trait_configs_and_target_node_dict=
            {
                'F': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=6), QuantizerConfig(bits=8)],
                      InsertionPointGraph.get_pre_hook_node_key('C')),
                'G': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=6), QuantizerConfig(bits=5,
                                                                mode=QuantizationMode.ASYMMETRIC)],
                      InsertionPointGraph.get_pre_hook_node_key('E')),
                'H': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=6), QuantizerConfig(bits=5,
                                                                mode=QuantizationMode.ASYMMETRIC)],
                      InsertionPointGraph.get_pre_hook_node_key('E')),
            },
            starting_master_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('D'),
            master_quantizer_qconfigs=[QuantizerConfig(bits=6)],
            target_node_for_master_quantizer=InsertionPointGraph.get_post_hook_node_key('A'),
            strategy_vs_expected_status_dict={
                PropagationStrategy.CONSERVATIVE: TransitionStatus.SHOULD_NOT_TRANSITION,
                PropagationStrategy.AGGRESSIVE: TransitionStatus.SHOULD_TRANSITION,
            }
        ),

        # Merge cases
        PathTransitionTestStruct(
            init_node_to_trait_configs_and_target_node_dict=
            {
                'E': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_pre_hook_node_key('A')),
                'D': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_pre_hook_node_key('A'))
            },
            starting_master_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('C'),
            master_quantizer_qconfigs=[QuantizerConfig()],
            target_node_for_master_quantizer=InsertionPointGraph.get_pre_hook_node_key('B'),
            strategy_vs_expected_status_dict={
                PropagationStrategy.CONSERVATIVE: TransitionStatus.SHOULD_MERGE,
                PropagationStrategy.AGGRESSIVE: TransitionStatus.SHOULD_MERGE,
            }
        ),

        PathTransitionTestStruct(
            init_node_to_trait_configs_and_target_node_dict=
            {
                'G': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_pre_hook_node_key('A')),
                'F': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=2)],
                      InsertionPointGraph.get_pre_hook_node_key('C'))
            },
            starting_master_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('H'),
            master_quantizer_qconfigs=[QuantizerConfig()],
            target_node_for_master_quantizer=InsertionPointGraph.get_pre_hook_node_key('B'),
            strategy_vs_expected_status_dict={
                PropagationStrategy.CONSERVATIVE: TransitionStatus.SHOULD_MERGE,
                PropagationStrategy.AGGRESSIVE: TransitionStatus.SHOULD_MERGE,
            }
        ),

        # No transition cases:

        # Path blocked by a quantizer
        PathTransitionTestStruct(
            init_node_to_trait_configs_and_target_node_dict=
            {
                'C': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_pre_hook_node_key('B')),
                'J': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=4)],
                      InsertionPointGraph.get_pre_hook_node_key('I')),
            },
            starting_master_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('F'),
            master_quantizer_qconfigs=[QuantizerConfig()],
            target_node_for_master_quantizer=InsertionPointGraph.get_pre_hook_node_key('C'),
            strategy_vs_expected_status_dict={
                PropagationStrategy.CONSERVATIVE: TransitionStatus.SHOULD_NOT_TRANSITION,
                PropagationStrategy.AGGRESSIVE: TransitionStatus.SHOULD_NOT_TRANSITION,
            }
        ),

        # Path blocked by a non-quantizable node
        PathTransitionTestStruct(
            init_node_to_trait_configs_and_target_node_dict=
            {
                'J': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_pre_hook_node_key('H')),
                'B': (QuantizationTrait.NON_QUANTIZABLE,
                      [],
                      None)
            },
            starting_master_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('C'),
            master_quantizer_qconfigs=[QuantizerConfig()],
            target_node_for_master_quantizer=InsertionPointGraph.get_pre_hook_node_key('A'),
            strategy_vs_expected_status_dict={
                PropagationStrategy.CONSERVATIVE: TransitionStatus.SHOULD_NOT_TRANSITION,
                PropagationStrategy.AGGRESSIVE: TransitionStatus.SHOULD_NOT_TRANSITION,
            }
        ),


        # A downward branch node was marked as non-quantizable
        PathTransitionTestStruct(
            init_node_to_trait_configs_and_target_node_dict=
            {
                'E': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_post_hook_node_key('B')),
                'D': (QuantizationTrait.NON_QUANTIZABLE,
                      [],
                      None)
            },
            starting_master_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('C'),
            master_quantizer_qconfigs=[QuantizerConfig()],
            target_node_for_master_quantizer=InsertionPointGraph.get_pre_hook_node_key('B'),
            strategy_vs_expected_status_dict={
                PropagationStrategy.CONSERVATIVE: TransitionStatus.SHOULD_NOT_TRANSITION,
                PropagationStrategy.AGGRESSIVE: TransitionStatus.SHOULD_NOT_TRANSITION,
            }
        ),

        # Incompatible upstream quantizer
        PathTransitionTestStruct(
            init_node_to_trait_configs_and_target_node_dict=
            {
                'E': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(6)],
                      InsertionPointGraph.get_post_hook_node_key('A')),
            },
            starting_master_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('D'),
            master_quantizer_qconfigs=[QuantizerConfig()],
            target_node_for_master_quantizer=InsertionPointGraph.get_pre_hook_node_key('B'),
            strategy_vs_expected_status_dict={
                PropagationStrategy.CONSERVATIVE: TransitionStatus.SHOULD_NOT_TRANSITION,
                PropagationStrategy.AGGRESSIVE: TransitionStatus.SHOULD_NOT_TRANSITION,
            }
        ),

        PathTransitionTestStruct(
            init_node_to_trait_configs_and_target_node_dict=
            {
                'E': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(6)],
                      InsertionPointGraph.get_post_hook_node_key('A')),
                'C': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(6)],
                      InsertionPointGraph.get_post_hook_node_key('A')),
            },
            starting_master_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('D'),
            master_quantizer_qconfigs=[QuantizerConfig()],
            target_node_for_master_quantizer=InsertionPointGraph.get_pre_hook_node_key('B'),
            strategy_vs_expected_status_dict={
                PropagationStrategy.CONSERVATIVE: TransitionStatus.SHOULD_NOT_TRANSITION,
                PropagationStrategy.AGGRESSIVE: TransitionStatus.SHOULD_NOT_TRANSITION,
            }
        ),

        # Incompatible downstream quantizers
        PathTransitionTestStruct(
            init_node_to_trait_configs_and_target_node_dict=
            {
                'F': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=6), QuantizerConfig(bits=8)],
                      InsertionPointGraph.get_pre_hook_node_key('C')),
                'G': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=6), QuantizerConfig(bits=5,
                                                                mode=QuantizationMode.ASYMMETRIC)],
                      InsertionPointGraph.get_pre_hook_node_key('E')),
                'H': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=6), QuantizerConfig(bits=5,
                                                                mode=QuantizationMode.ASYMMETRIC)],
                      InsertionPointGraph.get_pre_hook_node_key('E')),
            },
            starting_master_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('D'),
            master_quantizer_qconfigs=[QuantizerConfig(bits=4)],
            target_node_for_master_quantizer=InsertionPointGraph.get_post_hook_node_key('A'),
            strategy_vs_expected_status_dict={
                PropagationStrategy.CONSERVATIVE: TransitionStatus.SHOULD_NOT_TRANSITION,
                PropagationStrategy.AGGRESSIVE: TransitionStatus.SHOULD_NOT_TRANSITION,
            }
        ),

    ]

    @staticmethod
    @pytest.fixture(params=PATH_TRANSITION_TEST_CASES)
    def path_transition_test_struct(request):
        return request.param

    def test_check_transition_via_path(self, path_transition_test_struct: PathTransitionTestStruct):
        #pylint:disable=line-too-long
        init_node_to_trait_configs_and_target_node_dict = path_transition_test_struct.init_node_to_trait_configs_and_target_node_dict
        starting_master_quantizer_ip_node = path_transition_test_struct.starting_master_quantizer_ip_node
        master_quantizer_qconfigs = path_transition_test_struct.master_quantizer_qconfigs
        target_node = path_transition_test_struct.target_node_for_master_quantizer
        strategy_vs_status = path_transition_test_struct.strategy_vs_expected_status_dict

        # Graph preparation
        mock_graph = self.get_branching_model_graph()
        ip_graph = InsertionPointGraph(mock_graph)
        _, quant_prop_graph = self.prepare_propagation_graph_state(ip_graph,
                                                                   init_node_to_trait_configs_and_target_node_dict)

        master_prop_quant = quant_prop_graph.add_propagating_quantizer(master_quantizer_qconfigs,
                                                                       starting_master_quantizer_ip_node)
        path = get_edge_paths_for_propagation(quant_prop_graph,
                                              target_node,
                                              starting_master_quantizer_ip_node)[0]

        for strategy, ref_status in strategy_vs_status.items():
            solver = QuantizerPropagationSolver(propagation_strategy=strategy)
            status = solver.check_transition_via_path(master_prop_quant,
                                                      path,
                                                      quant_prop_graph)
            assert status == ref_status

    PropagationStepTestStruct = namedtuple('PropagationStepTestStruct',
                                           ('init_node_to_trait_configs_and_target_node_dict',
                                            'expected_finished_status',
                                            'current_location_node_key_for_propagated_quant'))
    PROPAGATION_STEP_TEST_CASES = [
        PropagationStepTestStruct(
            init_node_to_trait_configs_and_target_node_dict=
            {
                'F': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_pre_hook_node_key('A'))
            },
            expected_finished_status=True,
            current_location_node_key_for_propagated_quant=InsertionPointGraph.get_pre_hook_node_key('A')
        ),
        PropagationStepTestStruct(
            init_node_to_trait_configs_and_target_node_dict=
            {
                'F': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_pre_hook_node_key('C'))
            },
            expected_finished_status=False,
            current_location_node_key_for_propagated_quant=InsertionPointGraph.get_pre_hook_node_key('C')
        ),
        PropagationStepTestStruct(
            init_node_to_trait_configs_and_target_node_dict={
                'F': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_pre_hook_node_key('A')),
                'G': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_pre_hook_node_key('E')),
                'J': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_pre_hook_node_key('I'))
            },
            expected_finished_status=True,
            current_location_node_key_for_propagated_quant=InsertionPointGraph.get_pre_hook_node_key('A')
        )
    ]

    @staticmethod
    @pytest.fixture(params=PROPAGATION_STEP_TEST_CASES)
    def propagation_step_test_struct(request):
        return request.param

    def test_propagation_step(self, propagation_step_test_struct):
        # pylint:disable=line-too-long
        init_node_to_trait_configs_and_target_node_dict = propagation_step_test_struct.init_node_to_trait_configs_and_target_node_dict
        expected_finished_status = propagation_step_test_struct.expected_finished_status
        current_location_node_key_for_propagated_quant = propagation_step_test_struct.current_location_node_key_for_propagated_quant
        # Graph preparation
        mock_graph = self.get_branching_model_graph()
        ip_graph = InsertionPointGraph(mock_graph)
        quant_prop_solver = QuantizerPropagationSolver()
        # pylint:disable=line-too-long
        prop_quantizers, quant_prop_graph = self.prepare_propagation_graph_state(ip_graph,
                                                                                 init_node_to_trait_configs_and_target_node_dict)
        untouched_quantizers = []
        quant_prop = None
        for pq in prop_quantizers:
            if pq.current_location_node_key == current_location_node_key_for_propagated_quant:
                quant_prop = pq
            else:
                untouched_quantizers.append(pq)

        assert quant_prop is not None
        quant_prop_graph = quant_prop_solver.propagation_step(quant_prop, quant_prop_graph)

        if expected_finished_status:
            finished_propagating_quantizers = quant_prop_solver.get_finished_propagating_quantizers()
            assert quant_prop in finished_propagating_quantizers
        else:
            active_propagating_quantizers_queue = quant_prop_solver.get_active_propagating_quantizers_queue()
            assert quant_prop in active_propagating_quantizers_queue

        for pq in untouched_quantizers:
            assert not pq in quant_prop_solver.get_active_propagating_quantizers_queue()
            assert not pq in quant_prop_solver.get_finished_propagating_quantizers()

    RunOnIpGraphTestStruct = namedtuple('RunOnIpGraphTestStruct',
                                        ('list_ops',
                                         'expected_retval',
                                         'expected_count_finished_quant',
                                         'expected_count_active_quant'))

    RUN_ON_IP_GRAPH_TEST_CASES = [
        RunOnIpGraphTestStruct(
            list_ops=['conv2d', 'batch_norm'],
            expected_retval={},
            expected_count_finished_quant=0,
            expected_count_active_quant=0
        ),
        RunOnIpGraphTestStruct(
            list_ops=['conv2d', 'gelu', "conv2d"],
            expected_retval={
                InsertionInfo(OperationExecutionContext('conv2d', Scope(), 0, [None])): [QuantizerConfig()],
                InsertionInfo(OperationExecutionContext('gelu', Scope(), 0, [None])): [QuantizerConfig()]
            },
            expected_count_finished_quant=2,
            expected_count_active_quant=0
        )
    ]

    @staticmethod
    @pytest.fixture(params=RUN_ON_IP_GRAPH_TEST_CASES)
    def run_on_ip_graph_test_struct(request):
        return request.param

    def test_run_on_ip_graph(self, run_on_ip_graph_test_struct):
        expected_retval = run_on_ip_graph_test_struct.expected_retval
        expected_count_finished_quant = run_on_ip_graph_test_struct.expected_count_finished_quant
        expected_count_active_quant = run_on_ip_graph_test_struct.expected_count_active_quant

        # Graph preparation
        node_keys = run_on_ip_graph_test_struct.list_ops
        mock_graph = self.get_sequentially_connected_model_graph(node_keys)
        ip_graph = InsertionPointGraph(mock_graph)

        for node in ip_graph.nodes.values():
            if node[InsertionPointGraph.NODE_TYPE_NODE_ATTR] == InsertionPointGraphNodeType.OPERATOR:
                op_exec_context = node[InsertionPointGraph.REGULAR_NODE_REF_NODE_ATTR][
                    NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR]
                op_name = op_exec_context.operator_name
                ref_meta = OPERATOR_METATYPES.get_operator_metatype_by_op_name(op_name)
                node[InsertionPointGraph.OPERATOR_METATYPE_NODE_ATTR] = ref_meta

        quant_prop_solver = QuantizerPropagationSolver()
        retval = quant_prop_solver.run_on_ip_graph(ip_graph)

        assert retval == expected_retval

        assert len(quant_prop_solver.get_active_propagating_quantizers_queue()) == expected_count_active_quant
        assert len(quant_prop_solver.get_finished_propagating_quantizers()) == expected_count_finished_quant
