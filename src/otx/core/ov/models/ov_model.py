# type: ignore
# TODO: Need to remove line 1 (ignore mypy) and fix mypy issues
"""Modules for otx.core.ov.models.ov_model."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import math
import os
import tempfile
from collections import OrderedDict
from copy import deepcopy
from typing import Callable, List, Optional, Union

import openvino.runtime as ov
import torch
from torch.nn import init

from otx.utils.logger import get_logger

from ..graph import Graph
from ..graph.utils import (
    handle_merging_into_batchnorm,
    handle_paired_batchnorm,
    handle_reshape,
)
from ..ops.builder import OPS
from ..utils import load_ov_model, normalize_name

CONNECTION_SEPARATOR = "||"

# pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements
logger = get_logger()


class OVModel(torch.nn.Module):  # pylint: disable=too-many-instance-attributes
    """OVModel class."""

    def __init__(  # noqa: C901
        self,
        model_path_or_model: Union[str, ov.Model] = None,
        weight_path: Optional[str] = None,
        inputs: Optional[Union[str, List[str]]] = None,
        outputs: Optional[Union[str, List[str]]] = None,
        features_to_keep: Optional[List] = None,
        remove_normalize: bool = False,
        merge_bn: bool = True,
        paired_bn: bool = True,
        init_weight: Union[bool, Callable] = False,
        verify_shape: bool = True,
    ):
        super().__init__()
        self._model_path_or_model = model_path_or_model
        self._weight_path = weight_path
        self._remove_normalize = remove_normalize
        self._features_to_keep = features_to_keep
        self._merge_bn = merge_bn
        self._paired_bn = paired_bn
        self._init_weight = init_weight
        self._verify_shape = verify_shape

        self._inputs: List[str] = []
        self._outputs: List[str] = []
        self._feature_dict = OrderedDict()

        # build graph
        graph = self.build_graph(model_path_or_model, weight_path)
        self._graph = graph
        if remove_normalize:
            graph.remove_normalize_nodes()

        # handle inputs
        if inputs:
            inputs = inputs if isinstance(inputs, list) else [inputs]
            assert all(isinstance(i, str) for i in inputs), f"input must be string but {inputs} is given"
            inputs = self.build_custom_inputs(graph, deepcopy(inputs))
        else:
            inputs = [node.name for node in graph.get_nodes_by_types(["Parameter"])]
        self._inputs = inputs

        # handle outputs
        if outputs:
            outputs = outputs if isinstance(outputs, list) else [outputs]
            assert all(isinstance(i, str) for i in outputs), f"input must be string but {outputs} is given"
            outputs = self.build_custom_outputs(graph, deepcopy(outputs))
        else:
            outputs = [node.name for node in graph.get_nodes_by_types(["Result"])]
        self._outputs = outputs

        # clean up graph
        self.clean_up(graph, inputs, outputs)

        handle_reshape(graph)
        if merge_bn:
            handle_merging_into_batchnorm(graph)
        if paired_bn:
            handle_paired_batchnorm(graph, replace=True)

        # clean up graph
        self.clean_up(graph, inputs, outputs)

        # build torch module
        self.model = self.build_torch_module(graph)

        if init_weight:
            if not isinstance(init_weight, Callable):

                # internal init weight
                def init_weight(module, graph):  # pylint: disable=function-redefined
                    from ..ops.op import Operation

                    if not isinstance(module, Operation):
                        return

                    if module.TYPE == "BatchNormInference":
                        _, gamma, beta, mean, var = list(graph.predecessors(module))
                        init.ones_(gamma.data)
                        init.zeros_(beta.data)
                        mean.data.zero_()
                        var.data.fill_(1)
                        logger.info(f"Initialize {module.TYPE} -> {module.name}")
                    elif module.TYPE in [
                        "Convolution",
                        "GroupConvolution",
                        "MatMul",
                    ]:
                        for weight in graph.predecessors(module):
                            if weight.TYPE == "Constant" and isinstance(weight.data, torch.nn.parameter.Parameter):
                                init.kaiming_uniform_(weight.data, a=math.sqrt(5))
                                logger.info(f"Initialize {module.TYPE} -> {module.name}")
                    elif module.TYPE in [
                        "Multiply",
                        "Divide",
                        "Add",
                        "Subtract",
                    ]:
                        for weight in graph.predecessors(module):
                            if weight.TYPE == "Constant" and isinstance(weight.data, torch.nn.parameter.Parameter):
                                fan_in, _ = init._calculate_fan_in_and_fan_out(  # pylint: disable=protected-access
                                    weight.data
                                )
                                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                                init.uniform_(weight.data, -bound, bound)
                                logger.info(f"Initialize {module.TYPE} -> {module.name}")

            self.model.apply(lambda m: init_weight(m, graph))

        for node in self._graph.get_nodes_by_types(["Parameter"]):
            node.attrs.verify_shape = verify_shape

        input_shapes = {}
        output_shapes = {}
        for node in self._graph.get_nodes_by_types(["Parameter", "Result"]):
            if node.name in self._inputs:
                input_shapes[node.name] = node.shape[0]
            elif node.name in self._outputs:
                output_shapes[node.name] = node.shape[0]
        self._input_shapes = OrderedDict()
        self._output_shapes = OrderedDict()
        for input_ in self._inputs:
            self._input_shapes[input_] = input_shapes[input_]
        for output in self._outputs:
            self._output_shapes[output] = output_shapes[output]

    @property
    def inputs(self):
        """Property inputs."""
        return self._inputs

    @property
    def outputs(self):
        """Property outputs."""
        return self._outputs

    @property
    def features(self):
        """Property features."""
        return self._feature_dict

    @property
    def input_shapes(self):
        """Property input_shapes."""
        return self._input_shapes

    @property
    def output_shapes(self):
        """Property output_shapes."""
        return self._output_shapes

    @staticmethod
    def build_graph(model_path_or_model, weight_path=None):
        """Function build_graph."""
        with tempfile.TemporaryDirectory() as tempdir:
            if isinstance(model_path_or_model, ov.Model):
                assert weight_path is None, "if openvino model is given 'weight_path' must be None"
                ov.serialize(
                    model_path_or_model,
                    os.path.join(tempdir, "model.xml"),
                    os.path.join(tempdir, "model.bin"),
                )
                model_path_or_model = os.path.join(tempdir, "model.xml")
                weight_path = os.path.join(tempdir, "model.bin")
            # TODO: reshape decompose ir graph
            ov_model = load_ov_model(model_path_or_model, weight_path, False)
        graph = Graph.from_ov(ov_model)
        return graph

    @staticmethod
    def build_custom_outputs(graph, outputs):  # noqa: C901
        """Function build_custom_outputs."""
        cls_result = OPS.get_by_type_version("Result", "opset1")
        node_dict = OrderedDict((i.name, i) for i in graph.topological_sort())

        if not isinstance(outputs, list):
            outputs = [outputs]

        nodes_to_remove = []
        edges_to_add = {}
        for i, output in enumerate(outputs):
            output = normalize_name(output)
            output = output.split(CONNECTION_SEPARATOR)
            explicit_tgt = False

            if len(output) == 1:
                src = output[0]
                tgt = None
            elif len(output) == 2:
                src, tgt = output
                explicit_tgt = True
            else:
                raise ValueError()

            src = node_dict[src]
            if src.type == "Result":
                continue

            if explicit_tgt:
                tgt = node_dict[tgt]
            else:
                tgt = list(graph.successors(src))[0]

            output_result = f"{src.name}/result_{i}"
            outputs[i] = output_result

            if src not in edges_to_add:
                edges_to_add[src] = []

            for successor in graph.successors(src):
                if tgt == successor:
                    edges_attrs = graph.get_edge_data(src, successor)
                    assert len(edges_attrs) == 1

                    output_result = cls_result(output_result, shape=src.shape)
                    for edge_attrs in edges_attrs:
                        edges_to_add[src].append({"node_from": src, "node_to": output_result, **edge_attrs})
                if explicit_tgt and tgt != successor:
                    continue
                nodes_to_remove.append(successor)

        #  handle duplicated successors
        merge_candidates = [k for k, v in edges_to_add.items() if len(v) > 1]
        if merge_candidates:
            for merge_candidate in merge_candidates:
                edges = edges_to_add[merge_candidate]
                seen = {}
                out_ports = [edge["out_port"] for edge in edges]
                for idx in reversed(range(len(out_ports))):
                    out_port = out_ports[idx]
                    if out_port in seen:
                        edge = edges.pop(idx)
                        outputs.pop(outputs.index(edge["node_to"].name))
                    else:
                        seen[out_port] = edges[idx]

        if edges_to_add:
            for edges in edges_to_add.values():
                for edge in edges:
                    edge["in_port"] = 0
            assert {len(edges) for edges in edges_to_add.values()} == {1}
            edges_to_add = [edge for edges in edges_to_add.values() for edge in edges]
        else:
            edges_to_add = []

        for node in set(nodes_to_remove):
            graph.remove_node(node)
        for edge in edges_to_add:
            graph.add_edge(**edge)
        return outputs

    @staticmethod
    def build_custom_inputs(graph, inputs: Union[str, List[str]]):  # noqa: C901
        """Function build_custom_inputs."""
        cls_param = OPS.get_by_type_version("Parameter", "opset1")
        node_dict = OrderedDict((i.name, i) for i in graph.topological_sort())

        if not isinstance(inputs, list):
            inputs = [inputs]

        edges_to_add = {}
        nodes_to_remove = []
        for i, input_ in enumerate(inputs):
            input_ = normalize_name(input_)
            input_ = input_.split(CONNECTION_SEPARATOR)
            explicit_src = False

            if len(input_) == 1:
                src = None
                tgt = input_[0]
            elif len(input_) == 2:
                src, tgt = input_
                explicit_src = True
            else:
                raise ValueError()

            tgt = node_dict[tgt]
            if tgt.type == "Parameter":
                continue

            if explicit_src:
                src = node_dict[src]
            else:
                src = list(graph.predecessors(tgt))[0]

            input_parameter = f"{tgt.name}/parameter_{i}"
            inputs[i] = input_parameter

            if src not in edges_to_add:
                edges_to_add[src] = []

            for predecessor in graph.predecessors(tgt):
                if src == predecessor:
                    edges_attrs = graph.get_edge_data(predecessor, tgt)
                    assert len(edges_attrs) == 1

                    # TODO: here, we force the batch dim to be dynamic
                    # it is assumed to be dim 0
                    new_shape = []
                    for shape in predecessor.shape:
                        new_shape.append([-1 if j == 0 else k for j, k in enumerate(shape)])
                    new_shape = tuple(tuple(shape) for shape in new_shape)
                    input_parameter = cls_param(input_parameter, shape=new_shape)
                    for edge_attrs in edges_attrs:
                        edges_to_add[src].append({"node_from": input_parameter, "node_to": tgt, **edge_attrs})
                if (explicit_src and src != predecessor) or predecessor.type == "Constant":
                    continue
                nodes_to_remove.append(predecessor)

        # handle duplicated predecessors
        merge_candidates = [k for k, v in edges_to_add.items() if len(v) > 1]
        if merge_candidates:
            for merge_candidate in merge_candidates:
                ctr = 0
                edges = edges_to_add[merge_candidate]
                seen = {}
                out_ports = [edge["out_port"] for edge in edges]
                for idx in reversed(range(len(out_ports))):
                    out_port = out_ports[idx]
                    if out_port in seen:
                        edge = edges.pop(idx)
                        inputs.pop(inputs.index(edge["node_from"].name))
                        edge["node_from"] = seen[out_port]["node_from"]
                        edges_to_add[f"{merge_candidate.name}_{ctr}"] = [edge]
                        ctr += 1
                    else:
                        seen[out_port] = edges[idx]

        if edges_to_add:
            for edges in edges_to_add.values():
                for edge in edges:
                    edge["out_port"] = 0
            assert {len(edges) for edges in edges_to_add.values()} == {1}
            edges_to_add = [edge for edges in edges_to_add.values() for edge in edges]
        else:
            edges_to_add = []

        for node in set(nodes_to_remove):
            graph.remove_node(node)
        for edge in edges_to_add:
            graph.add_edge(**edge)
        return inputs

    @staticmethod
    def clean_up(graph, inputs=None, outputs=None):
        """Function clean_up."""
        inputs = inputs if inputs else []
        outputs = outputs if outputs else []
        nodes = list(graph.topological_sort())
        nodes_to_keep = []
        for node in nodes:
            if node.name in inputs or node.name in outputs:
                nodes_to_keep.append(node)

        def get_nodes_without_successors(graph, ignores=None):
            ignores = ignores if ignores else []
            outputs = []
            for node in reversed(list(graph.topological_sort())):
                if not list(graph.successors(node)) and node not in ignores:
                    outputs.append(node)
            return outputs

        nodes = get_nodes_without_successors(graph, nodes_to_keep)
        while nodes:
            graph.remove_nodes_from(nodes)
            nodes = get_nodes_without_successors(graph, nodes_to_keep)

        graph.clean_up(nodes_to_keep)

    @staticmethod
    def build_torch_module(graph):
        """Function build_torch_module."""
        node_dict = OrderedDict((i.name, i) for i in graph.topological_sort())
        return torch.nn.ModuleDict(list(node_dict.items()))

    def _build_forward_inputs(self, *args, **kwargs):
        """Function _build_forward_inputs."""
        inputs = {}
        if args:
            for key, arg in zip(self._inputs, args):
                inputs[key] = arg
        if kwargs:
            for key, arg in kwargs.items():
                if key in inputs:
                    raise ValueError
                inputs[key] = arg
        return inputs

    def forward(self, *args, **kwargs):
        """Function forward."""
        self._feature_dict.clear()
        inputs = self._build_forward_inputs(*args, **kwargs)

        done = {}
        for node_name, node in self.model.items():
            done[node_name] = {node.name: False for node in self._graph.successors(node)}

        for node_name, node in self.model.items():
            predecessors_with_edge = list(self._graph.predecessors(node, with_edge_data=True))
            if not predecessors_with_edge:
                if node.type == "Parameter":
                    self._feature_dict[node_name] = node(inputs[node_name])
                elif node.type == "Constant":
                    self._feature_dict[node_name] = node()
                else:
                    raise ValueError(
                        f"Broken graph. Node {node_name} is a type of {node.type} " "but it has no in edges."
                    )
            else:
                input_nodes, edges = list(map(list, zip(*predecessors_with_edge)))
                input_node_names = [input_node.name for input_node in input_nodes]

                input_features = [edge["in_port"] for edges_ in edges for edge in edges_]
                assert len(input_features) == len(set(input_features))
                input_features = [None for _ in input_features]
                for idx, input_node_name in enumerate(input_node_names):
                    if self._features_to_keep is not None and input_node_name in self._features_to_keep:
                        input_feature = self._feature_dict.get(input_node_name)
                    else:
                        input_feature = self._feature_dict.pop(input_node_name)
                        done[input_node_name][node_name] = True
                        if not all(done[input_node_name].values()):
                            self._feature_dict[input_node_name] = input_feature

                    if isinstance(input_feature, tuple):
                        for edges_ in edges[idx]:
                            input_features[edges_["in_port"]] = input_feature[edges_["out_port"]]
                    else:
                        for edges_ in edges[idx]:
                            input_features[edges_["in_port"]] = input_feature
                assert all(input_feature is not None for input_feature in input_features)
                self._feature_dict[node_name] = node(*input_features)

        outputs = OrderedDict()
        for output_name in self._outputs:
            outputs[output_name] = self._feature_dict[output_name]

        return outputs
