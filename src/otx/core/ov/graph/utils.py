"""Utils for otx.core.ov.graph."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, List

import torch

from otx.core.ov.graph import Graph
from otx.core.ov.ops.builder import OPS
from otx.core.ov.ops.infrastructures import ConstantV0
from otx.core.ov.ops.op import Operation
from otx.utils.logger import get_logger

# pylint: disable=too-many-locals, protected-access, too-many-branches, too-many-statements, too-many-nested-blocks
logger = get_logger()


def get_constant_input_nodes(graph: Graph, node: Operation) -> List[Operation]:
    """Getter constant input nodes from graph, node."""
    found = []
    for node_ in graph.predecessors(node):
        if node_.type == "Constant":
            found.append(node_)
    return found


def handle_merging_into_batchnorm(graph, type_patterns=None, type_mappings=None):  # noqa: C901
    """Merge function graph into batchnorm."""
    type_patterns = type_patterns if type_patterns else [["Multiply", "Add"]]
    type_mappings = type_mappings if type_mappings else [{"gamma": 0, "beta": 1}]
    assert len(type_patterns) == len(type_mappings)
    batchnorm_cls = OPS.get_by_type_version("BatchNormInference", "opset1")
    constant_cls = OPS.get_by_type_version("Constant", "opset1")

    for node in list(graph.nodes.keys()):
        if node not in graph:
            continue
        nodes = []
        type_mapping = {}
        for pattern_idx, type_pattern in enumerate(type_patterns):
            type_mapping = type_mappings[pattern_idx]
            nodes = graph.get_nodes_by_type_pattern(type_pattern, node)
            if nodes and len(nodes) == 1:
                nodes = nodes[0]
                break
        if not nodes:
            continue

        is_normalize = False
        for normalize_nodes in graph._normalize_nodes:
            if set(nodes).intersection(normalize_nodes):
                is_normalize = True
                break
        if is_normalize:
            logger.info(
                f"Skip merging {[i.name for i in nodes]} "
                f"becuase they are part of normalization (preprocessing of IR)"
            )
            continue

        shapes = []
        constants = []
        is_valid = True
        for node in nodes:
            constant = get_constant_input_nodes(graph, node)
            if len(constant) != 1:
                is_valid = False
                break
            constant = constant[0]
            shapes.append(constant.shape)
            constants.append(constant)
        if not is_valid:
            logger.info(
                f"Skip merging {[i.name for i in nodes]} " f"becuase it has more than one weights for node {node.name}."
            )
            continue

        if len(set(shapes)) != 1:
            logger.info(
                f"Skip merging {[i.name for i in nodes]} " f"becuase shape of weights are not the same. ({shapes})"
            )
            continue

        if len(set(shapes[0][2:])) != 1 or shapes[0][2] != 1:
            logger.info(f"Skip merging {[i.name for i in nodes]} " f"becuase shape of weights are not 1. ({shapes})")
            continue

        channel_dim = shapes[0][1]

        name = nodes[0].name + "/merged_bn"
        batchnorm = batchnorm_cls(name, shape=node.shape, epsilon=1e-10)

        gamma = (
            constants[type_mapping["gamma"]].data.squeeze() if "gamma" in type_mapping else torch.ones([channel_dim])
        )
        gamma = constant_cls(
            gamma,
            batchnorm.name + "/gamma",
            shape=((channel_dim,),),
            is_parameter=True,
        )

        beta = constants[type_mapping["beta"]].data.squeeze() if "beta" in type_mapping else torch.zeros([channel_dim])
        beta = constant_cls(
            beta,
            batchnorm.name + "/beta",
            shape=((channel_dim,),),
            is_parameter=True,
        )

        running_mean = (
            constants[type_mapping["running_mean"]].data.squeeze()
            if "running_mean" in type_mapping
            else torch.zeros([channel_dim])
        )
        running_mean = constant_cls(
            running_mean,
            batchnorm.name + "/running_mean",
            shape=((channel_dim,),),
            is_parameter=False,
        )

        running_variance = (
            constants[type_mapping["running_variance"]].data.squeeze()
            if "running_variance" in type_mapping
            else torch.ones([channel_dim])
        )
        running_variance = constant_cls(
            running_variance,
            batchnorm.name + "/running_variance",
            shape=((channel_dim,),),
            is_parameter=False,
        )

        logger.info(f"Merge {[i.name for i in nodes]} into batch normalization.")
        edges = []
        for predecessor in graph.predecessors(nodes[0]):
            if predecessor.type != "Constant":
                edges_attrs = graph.get_edge_data(predecessor, nodes[0])
                assert len(edges_attrs) == 1
                for edge_attrs in edges_attrs:
                    edges.append({"node_from": predecessor, "node_to": batchnorm, **edge_attrs})
        for successor in graph.successors(nodes[-1]):
            edges_attrs = graph.get_edge_data(nodes[-1], successor)
            assert len(edges_attrs) == 1
            for edge_attrs in edges_attrs:
                edges.append({"node_from": batchnorm, "node_to": successor, **edge_attrs})
        for node in nodes:
            graph.remove_node(node)
        for edge in edges:
            graph.add_edge(**edge)
        graph.add_edge(gamma, batchnorm)
        graph.add_edge(beta, batchnorm)
        graph.add_edge(running_mean, batchnorm)
        graph.add_edge(running_variance, batchnorm)


def handle_paired_batchnorm(graph, replace: bool = False, types: List[str] = None):
    """Handle function paired batchnorm."""
    types = types if types else ["Convolution", "GroupConvolution"]
    batchnorm_cls = OPS.get_by_type_version("BatchNormInference", "opset1")
    constant_cls = OPS.get_by_type_version("Constant", "opset1")

    for node in list(graph.nodes.keys()):
        if node.type not in types:
            continue

        # if input is 1x1x... this node is probably in Squeeze-and-exitation network
        input_node, edge = list(graph.predecessors(node, True))[0]
        assert len(edge) == 1
        edge = edge[0]
        input_shape = input_node.shape[edge["out_port"]][2:]
        if len(set(input_shape)) == 1 and input_shape[0] == 1:
            logger.info(
                f"Skip a paired batch normalization for {node.name} " f"becuase input shape to it is {input_shape}."
            )
            continue

        bias_node_list: List[Any] = [n for n in graph.successors(node) if n.type == "Add"]
        if len(bias_node_list) == 1:
            bias_node = bias_node_list[0]
        else:
            bias_node = None

        # if bias node is not found we do not need to add batchnorm
        if bias_node is None:
            logger.info(f"Skip a paired batch normalization for {node.name} " "becuase it has no bias add node.")
            continue
        # if add node is not bias add node
        if not isinstance(list(graph.predecessors(bias_node))[1], ConstantV0):
            logger.info(
                f"Skip a pared batch normalization for {node.name} " f"because {bias_node.name} is not a bias add node."
            )
            continue

        node_name = node.name
        channel_dim = node.attrs.shape[0][1]

        batchnorm = batchnorm_cls(node_name + "/paried_bn", shape=node.shape, epsilon=1e-10)

        gamma = torch.ones([channel_dim])
        gamma = constant_cls(
            batchnorm.name + "/gamma",
            data=gamma,
            shape=((channel_dim,),),
            is_parameter=True,
        )
        if replace and bias_node is not None:
            beta = list(graph.predecessors(bias_node))[1].data.squeeze()
        else:
            beta = torch.zeros([channel_dim])
        beta = constant_cls(
            batchnorm.name + "/beta",
            data=beta,
            shape=((channel_dim,),),
            is_parameter=True,
        )
        running_mean = torch.zeros([channel_dim])
        running_mean = constant_cls(
            batchnorm.name + "/running_mean",
            data=running_mean,
            shape=((channel_dim,),),
            is_parameter=False,
        )
        running_variance = torch.ones([channel_dim])
        running_variance = constant_cls(
            batchnorm.name + "/running_variance",
            data=running_variance,
            shape=((channel_dim,),),
            is_parameter=False,
        )

        if replace and bias_node is not None:
            logger.info(f"Replace {bias_node.name} with a paired batch normalization.")
            edges = []
            for successor in graph.successors(bias_node):
                edges_attrs = graph.get_edge_data(bias_node, successor)
                assert len(edges_attrs) == 1
                for edge_attrs in edges_attrs:
                    edges.append({"node_from": batchnorm, "node_to": successor, **edge_attrs})
            for predecessor in graph.predecessors(bias_node):
                if predecessor.type != "Constant":
                    edges_attrs = graph.get_edge_data(predecessor, bias_node)
                    assert len(edges_attrs) == 1
                    for edge_attrs in edges_attrs:
                        edges.append(
                            {
                                "node_from": predecessor,
                                "node_to": batchnorm,
                                **edge_attrs,
                            }
                        )
            graph.remove_node(bias_node)
            for edge in edges:
                graph.add_edge(**edge)
        else:
            logger.info(f"Append a paired batch normalization after {node.name}")
            edges = []
            for successor in graph.successors(node):
                edges_attrs = graph.get_edge_data(node, successor)
                assert len(edges_attrs) == 1
                for edge_attrs in edges_attrs:
                    edges.append({"node_from": batchnorm, "node_to": successor, **edge_attrs})
                graph.remove_edge(node, successor)
            for edge in edges:
                graph.add_edge(**edge)
            graph.add_edge(node, batchnorm)

        graph.add_edge(gamma, batchnorm)
        graph.add_edge(beta, batchnorm)
        graph.add_edge(running_mean, batchnorm)
        graph.add_edge(running_variance, batchnorm)


def handle_reshape(graph):
    """Reshape function."""
    for result in graph.get_nodes_by_types(["Result"]):
        for node in graph.predecessors(result):
            # some models, for example, dla-34, have reshape node as its predecessor
            # of result node and the reshape node reshapes the tensor to [1, -1]
            if node.type == "Reshape":
                input_node, shape = list(graph.predecessors(node))
                if torch.equal(shape.data, torch.tensor([1, -1])):
                    for shape_ in input_node.shape[0][::-1]:
                        if shape_ != 1:
                            break
                    logger.info(f"Change reshape to [-1, {shape_}]")  # pylint: disable=undefined-loop-variable
                    shape.data = torch.tensor([-1, shape_])  # pylint: disable=undefined-loop-variable
