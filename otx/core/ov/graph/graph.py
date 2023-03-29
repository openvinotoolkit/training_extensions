# type: ignore
# TODO: Need to remove line 1 (ignore mypy) and fix mypy issues
"""Modules for otx.core.ov.graph."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import inspect
from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import _collections_abc
import networkx as nx
from openvino.pyopenvino import Model  # pylint: disable=no-name-in-module

from otx.algorithms.common.utils.logger import get_logger

from ..ops.op import Operation
from ..ops.utils import convert_op_to_torch
from ..utils import get_op_name

# pylint: disable=too-many-locals, too-many-nested-blocks, arguments-renamed, too-many-branches, too-many-statements

logger = get_logger()


class SortedDictKeysView(_collections_abc.KeysView):
    """SortedDictKeysView class."""

    def __repr__(self):
        """Function repr of SortedDictKeysView."""
        return f"{self.__class__.__name__}({list(self._mapping)})"

    def __reversed__(self):
        """Function reversed of SortedDictKeysView."""
        yield from reversed(self._mapping)


class SortedDictValuesView(_collections_abc.ValuesView):
    """SortedDictValuesView class."""

    def __repr__(self):
        """Sorteddictvaluesview's repr function."""
        return f"{self.__class__.__name__}({[self._mapping[i] for i in self._mapping]})"

    def __reversed__(self):
        """Sorteddictvaluesview's reversed function."""
        for key in reversed(self._mapping):
            yield self._mapping[key]


class SortedDictItemsView(_collections_abc.ItemsView):
    """SortedDictItemsView class."""

    def __repr__(self):
        """Sorteddictitemsview's repr function."""
        return f"{self.__class__.__name__}({[(i, self._mapping[i]) for i in self._mapping]})"

    def __reversed__(self):
        """Sorteddictitemsview's reversed function."""
        for key in reversed(self._mapping):
            yield (key, self._mapping[key])


class NOOP:
    """NOOP class."""

    pass  # pylint: disable=unnecessary-pass


class SortedDict(dict):
    """SortedDict class."""

    def __init__(self, sort_key, *args, **kwargs):
        self._sort_key = sort_key
        self._sorted_keys = []
        super().__init__(self, *args, **kwargs)

    def __setitem__(self, key, value):
        """Sorteddict's setitem function."""
        assert len(value) == 1
        edge_key, edge_attr = next(iter(value.items()))
        sort_value = float("inf") if self._sort_key not in edge_attr else edge_attr[self._sort_key]
        self._sorted_keys.append([sort_value, key, edge_key])
        self._sorted_keys.sort(key=lambda x: x[0])
        if key in self:
            assert edge_key not in self[key]
            self[key].update(value)
        else:
            super().__setitem__(key, value)

    def __delitem__(self, key):
        """Sorteddict's delitem function."""
        super().__delitem__(key)
        for i, (_, key_in, _) in enumerate(self._sorted_keys):
            if key_in == key:
                break
        self._sorted_keys.pop(i)  # pylint: disable=undefined-loop-variable

    def __iter__(self):
        """Sorteddict's iter function."""
        for _, key, _ in self._sorted_keys:
            yield key

    def __reversed__(self):
        """Sorteddict's reversed function."""
        for _, key, _ in self._sorted_keys[::-1]:
            yield key

    def __repr__(self):
        """Sorteddict's repr function."""
        if not len(self):  # pylint: disable=use-implicit-booleaness-not-len
            return "{}"
        repr_ = "{"
        for _, key, _ in self._sorted_keys:
            repr_ += f"{key}: {self[key]}, "
        repr_ = repr_[:-2]
        repr_ += "}"
        return repr_

    def __deepcopy__(self, memo):
        """Sorteddict's deepcopy function."""
        cls = self.__class__
        result = cls(self._sort_key)
        memo[id(self)] = result
        for key, value in self.items():
            result[key] = deepcopy(value, memo)
        return result

    def clear(self):
        """Sorteddict's clear function."""
        super().clear()
        self._sorted_keys = []

    def pop(self, key, default=NOOP()):
        """Sorteddict's pop function."""
        if isinstance(default, NOOP):
            value = super().pop(key)
        else:
            value = super().pop(key, default)

        for i, (_, key_in, _) in enumerate(self._sorted_keys):
            if key_in == key:
                break
        self._sorted_keys.pop(i)  # pylint: disable=undefined-loop-variable

        return value

    def popitem(self):
        """Sorteddict's popitem function."""
        raise NotImplementedError

    @staticmethod
    def fromkeys(iterable, value=None):
        """Sorteddict's fromkeys function."""
        raise NotImplementedError

    def keys(self):
        """Sorteddict's keys function."""
        return SortedDictKeysView(self)

    def values(self):
        """Sorteddict's values function."""
        return SortedDictValuesView(self)

    def items(self):
        """Sorteddict's items function."""
        return SortedDictItemsView(self)


class SortedDictHelper(dict):
    """SortedDictHelper class."""

    def __init__(self, sort_key=None, *args, **kwargs):  # pylint: disable=keyword-arg-before-vararg
        self._sort_key = sort_key
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        """Sorteddicthelper's setitem function."""
        super().__setitem__(key, SortedDict(self._sort_key))
        for v_key, v_value in value.items():
            self[key][v_key] = v_value


class Graph(nx.MultiDiGraph):
    """Graph class."""

    adjlist_outer_dict_factory = SortedDictHelper

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._adj = self.adjlist_outer_dict_factory("out_port")
        self._pred = self.adjlist_outer_dict_factory("in_port")
        self._succ = self._adj

        self._normalize_nodes = []

    @staticmethod
    def from_ov(ov_model: Model) -> "Graph":
        """Graph's from_ov function."""
        graph = Graph()

        ov_ops = ov_model.get_ordered_ops()
        ops_dict = OrderedDict()
        parents_dict: Dict[str, List[Optional[List]]] = {}
        children_dict: Dict[str, List[Optional[List]]] = {}

        for ov_op in ov_ops:
            op_name = get_op_name(ov_op)
            node = convert_op_to_torch(ov_op)

            graph.add_node(
                node,
                name=op_name,
                type=node.type,
                version=node.version,
                attrs=node.attrs,
            )
            ops_dict[op_name] = node

            children_dict[op_name] = []
            for out_port in ov_op.outputs():
                out_port_id = out_port.get_index()
                for in_port in out_port.get_target_inputs():
                    in_port_id = in_port.get_index()
                    children_dict[op_name].append([out_port_id, in_port_id, get_op_name(in_port.get_node())])

            parents_dict[op_name] = []
            for in_port in ov_op.inputs():
                in_port_id = in_port.get_index()
                out_port = in_port.get_source_output()
                out_port_id = out_port.get_index()
                parents_dict[op_name].append([out_port_id, in_port_id, get_op_name(out_port.get_node())])

        # validate graph
        for node, children in children_dict.items():
            for _, _, child in children:
                assert node in [i[-1] for i in parents_dict[child]], f"{node} is not a parent of {child}"
        for node, parents in parents_dict.items():
            for _, _, parent in parents:
                assert node in [i[-1] for i in children_dict[parent]], f"{node} is not a child of {parent}"

        # add edges
        for src, tgts in children_dict.items():
            for out_port_id, in_port_id, tgt in tgts:
                graph.add_edge(
                    ops_dict[src],
                    ops_dict[tgt],
                    in_port=in_port_id,
                    out_port=out_port_id,
                )

        # freeze normalization nodes
        graph._freeze_normalize_nodes()  # pylint: disable=protected-access

        return graph

    def get_edge_data(self, node_from: Operation, node_to: Operation, default=None) -> Optional[List[Dict[Any, Any]]]:
        """Graph's get_edge_data function."""
        edge_data = super().get_edge_data(node_from, node_to, None, default)
        if edge_data is not None:
            return list(edge_data.values())
        return None

    def remove_node(self, node: Operation, keep_connect: bool = False):
        """Graph's remove_node function."""
        edges_to_keep = []
        if keep_connect:
            predecessors = [
                predecessor
                for predecessor in self.predecessors(node)
                if hasattr(predecessor, "type") and predecessor.type != "Constant"
            ]
            if predecessors:
                assert len(predecessors) == 1
                predecessor = predecessors[0]

                for successor in self.successors(node):
                    for predecessor_ in self.predecessors(successor):
                        edges_attrs = self.get_edge_data(predecessor_, successor)
                        assert len(edges_attrs) == 1
                        if predecessor_ == node:
                            for edge_attrs in edges_attrs:
                                edges_to_keep.append([predecessor, successor, edge_attrs])

        super().remove_node(node)
        for edge in edges_to_keep:
            node_from, node_to, attrs = edge
            self.add_edge(node_from, node_to, **attrs)

    def replace_node(self, old_node: Operation, new_node: Operation):
        """Graph's replace_node function."""
        edges = []
        for successor in self.successors(old_node):
            for edge_attrs in self.get_edge_data(old_node, successor):
                edges.append((new_node, successor, edge_attrs))
        for predecessor in self.predecessors(old_node):
            for edge_attrs in self.get_edge_data(predecessor, old_node):
                edges.append((predecessor, new_node, edge_attrs))

        self.remove_node(old_node)

        for edge in edges:
            node_from, node_to, attrs = edge
            self.add_edge(node_from, node_to, **attrs)

    def add_edge(
        self,
        node_from: Operation,
        node_to: Operation,
        out_port: Optional[int] = None,
        in_port: Optional[int] = None,
        **kwargs,
    ):
        """Graph's add_edge function."""
        if node_from not in self:
            self.add_node(node_from)

        if node_to not in self:
            self.add_node(node_to)

        if out_port is None:
            out_port = 0

        if in_port is None:
            occupied = [
                edge["in_port"]
                for predecessor in self.predecessors(node_to)
                for edge in self.get_edge_data(predecessor, node_to)
            ]
            assert len(occupied) == len(set(occupied))
            if occupied:
                for i in range(max(occupied)):
                    if i not in occupied:
                        in_port = i
                        break
            if in_port is None:
                in_port = len(occupied)

        # validate in_port
        spec = inspect.getfullargspec(node_to.forward)
        if spec.varargs is None:
            valid_range = list(range(len(spec.args[1:])))
            if in_port not in valid_range:
                raise ValueError(f"in_port {in_port} is not in valid range {valid_range} " f"for {node_to.name}.")
        occupied = []
        predecessors = list(self.predecessors(node_to))
        if predecessors:
            occupied = [
                edge["in_port"] for predecessor in predecessors for edge in self.get_edge_data(predecessor, node_to)
            ]
            assert len(occupied) == len(set(occupied))
        if occupied:
            if in_port in occupied:
                raise ValueError(f"in_port {in_port} is occupied for {node_to.name}.")

        # out_port validation is not able to do

        # add edge
        key = f"{in_port}{out_port}"
        super().add_edge(node_from, node_to, key=key, in_port=in_port, out_port=out_port, **kwargs)

    def predecessors(
        self,
        node: Operation,
        with_edge_data: bool = False,
    ) -> Generator[Union[Tuple[Operation, Optional[List]], Operation], None, None]:
        """Graph's predecessors function."""
        for predecessor in super().predecessors(node):
            if with_edge_data:
                yield (predecessor, self.get_edge_data(predecessor, node))
            else:
                yield predecessor

    def successors(
        self,
        node: Operation,
        with_edge_data: bool = False,
    ) -> Generator[Union[Tuple[Operation, Optional[List]], Operation], None, None]:
        """Graph's successors function."""
        for successor in super().successors(node):
            if with_edge_data:
                yield (successor, self.get_edge_data(node, successor))
            else:
                yield successor

    def get_nodes_by_types(self, types: List[str]) -> List[Operation]:
        """Graph's get_nodes_by_types function."""
        found = []
        for node in self.topological_sort():
            if node.type in types:
                found.append(node)
        return found

    def bfs(
        self, node: Operation, reverse: bool = False, depth_limit: Optional[int] = None
    ) -> Generator[Union[Tuple[Operation, Operation], Tuple[Operation, Tuple[Operation]]], None, None]:
        """Graph's bfs function."""
        if reverse:
            for s_value, t_value in nx.bfs_edges(self, node, reverse=True, depth_limit=depth_limit):
                yield (t_value, s_value)
        else:
            parent = node
            children = []
            for p, c in nx.bfs_edges(self, node, depth_limit=depth_limit):
                if p == parent:
                    children.append(c)
                    continue
                yield (parent, tuple(children))
                children = [c]
                parent = p
            if children:
                yield (parent, tuple(children))

    #  def dfs(self, node: Operation, forward=True, depth_limit=None):
    #      if forward:
    #          return nx.dfs_successors(self, node, depth_limit)
    #      else:
    #          return nx.dfs_predecessors(self, node, depth_limit)

    def get_nodes_by_type_pattern(self, pattern: List[str], start_node: Optional[Operation] = None, reverse=False):
        """Graph's get_nodes_by_type_pattern function."""
        if len(pattern) < 1:
            raise ValueError(f"pattern must be longer than 2 but {len(pattern)} is given")
        pattern_pairs = [pattern[i : i + 2] for i in range(len(pattern) - 1)]

        if start_node is None:
            if reverse:
                start_node = list(self.topological_sort())[-1]
            else:
                start_node = list(self.topological_sort())[0]

        founds = []
        start_nodes = [start_node]
        for pattern_pair in pattern_pairs:
            found_ = {start_node: None for start_node in start_nodes}
            for start_node_ in start_nodes:
                for s_value, ts_ in self.bfs(start_node_, reverse, 1):
                    if not isinstance(ts_, tuple):
                        ts_ = (ts_,)
                    for t in ts_:
                        if [s_value.type, t.type] == pattern_pair:
                            if reverse:
                                found_[t] = s_value
                            else:
                                found_[s_value] = t
            if founds:
                pop_indices = []
                for i, found in enumerate(founds):
                    last_node = found[-1]
                    if last_node not in found_.keys():
                        pop_indices.append(i)
                    else:
                        founds[i].append(found_[last_node])
                for pop_idx in pop_indices[::-1]:
                    founds.pop(pop_idx)
            else:
                founds = [[s, t] for s, t in found_.items() if s is not None and t is not None]
            start_nodes = [found[-1] for found in founds]
        return founds

    def _freeze_normalize_nodes(self):  # noqa: C901
        """Graph's _freeze_normalize_nodes function."""
        invariant_types = ["Transpose", "Convert"]

        def test_constant(node):
            """Graph's test_constant function."""
            constant_nodes = [node_ for node_ in self.predecessors(node) if node_.type == "Constant"]
            if len(constant_nodes) != 1:
                return False
            constant_value = constant_nodes[0].data.squeeze()
            if constant_value.dim() > 1 or constant_value.numel() not in [1, 3]:
                return False
            return True

        def get_nodes_by_type_from_node(
            node,
            types,
            ignore_types=None,
            reverse=False,
            depth_limit=-1,
        ):
            """Graph's get_nodes_by_type_from_node function."""
            ignore_types = ignore_types if ignore_types else []
            func = self.successors
            if reverse:
                func = self.predecessors

            candidates = [(i, 1) for i in func(node)]
            found = []
            for candidate, cur_depth in candidates:
                if cur_depth > depth_limit > -1:
                    break
                if candidate.type == types:
                    found.append(candidate)
                elif candidate.type in ignore_types:
                    candidates.extend([(i, cur_depth + 1) for i in func(candidate)])
            return found

        def find_multiply_add(node):
            """Graph's find_multiply_add function."""
            scale_node = None
            mean_node = None

            scale_nodes = get_nodes_by_type_from_node(node, "Multiply", invariant_types)
            if scale_nodes and len(scale_nodes) == 1:
                scale_node = scale_nodes[0]
                if not test_constant(scale_node):
                    scale_node = None

            node = scale_node if scale_node is not None else node
            mean_nodes = get_nodes_by_type_from_node(node, "Add", invariant_types)
            if mean_nodes and len(mean_nodes) == 1:
                mean_node = mean_nodes[0]
                if not test_constant(mean_node):
                    mean_node = None

            return (scale_node, mean_node)

        def find_subtract_divide(node):
            """Graph's find_subtract_divide function."""
            mean_node = None
            scale_node = None

            mean_nodes = get_nodes_by_type_from_node(node, "Subtract", invariant_types)
            if mean_nodes and len(mean_nodes) == 1:
                mean_node = mean_nodes[0]
                if not test_constant(mean_node):
                    mean_node = None

            node = mean_node if mean_node is not None else node
            scale_nodes = get_nodes_by_type_from_node(node, "Divide", invariant_types)
            if scale_nodes and len(scale_nodes) == 1:
                scale_node = scale_nodes[0]
                if not test_constant(scale_node):
                    scale_node = None

            return (mean_node, scale_node)

        def find_subtract_multiply(node):
            """Graph's find_subtract_multiply function."""
            mean_node = None
            scale_node = None

            mean_nodes = get_nodes_by_type_from_node(node, "Subtract", invariant_types)
            if mean_nodes and len(mean_nodes) == 1:
                mean_node = mean_nodes[0]
                if not test_constant(mean_node):
                    mean_node = None

            node = mean_node if mean_node is not None else node
            scale_nodes = get_nodes_by_type_from_node(node, "Multiply", invariant_types)
            if scale_nodes and len(scale_nodes) == 1:
                scale_node = scale_nodes[0]
                if not test_constant(scale_node):
                    scale_node = None

            return (mean_node, scale_node)

        for node in self:
            if node.type != "Parameter":
                continue

            # others
            found = find_multiply_add(node)
            if not any(found):
                # onnx, paddle
                found = find_subtract_divide(node)
                found_ = find_subtract_multiply(node)
                if len([i for i in found if i is not None]) < len([i for i in found_ if i is not None]):
                    found = found_

            if not all(i is not None for i in found):
                continue

            self._normalize_nodes.append(found)
            for normalize_node in found:
                if normalize_node is not None:
                    constant_node = [node_ for node_ in self.predecessors(normalize_node) if node_.type == "Constant"][
                        0
                    ]
                    attrs = constant_node.attrs
                    attrs = asdict(attrs)
                    attrs["is_parameter"] = False
                    new_constant_node = constant_node.__class__(
                        constant_node.name,
                        data=constant_node.data.data,
                        **attrs,
                    )
                    self.replace_node(constant_node, new_constant_node)

    def remove_normalize_nodes(self):
        """Graph's remove_normalize_nodes function."""
        for nodes in self._normalize_nodes:
            first_node, second_node = nodes

            if first_node is None:
                first_node = second_node
            elif second_node is None:
                second_node = first_node
            self.remove_node(first_node, keep_connect=True)
            logger.info(f"Remove normalize node {first_node.name}")
            try:
                self.remove_node(second_node, keep_connect=True)
                logger.info(f"Remove normalize node {second_node.name}")
            except Exception:  # pylint: disable=broad-exception-caught
                pass
        self._normalize_nodes = []

    def topological_sort(self):
        """Graph's topological_sort function."""
        return nx.topological_sort(self)

    def has_path(self, node_from: Operation, node_to: Operation):
        """Graph's has_path function."""
        return nx.has_path(self, node_from, node_to)

    def clean_up(
        self,
        nodes_to_keep: List[Operation] = None,
        remove_sub_components: bool = True,
    ):
        """Graph's clean_up function."""
        nodes_to_keep = nodes_to_keep if nodes_to_keep else []
        if remove_sub_components:
            # clean up sub components
            components = list(nx.connected_components(self.to_undirected()))
            if nodes_to_keep:
                for component in components:
                    if not set(nodes_to_keep).intersection(component):
                        super().remove_nodes_from(list(component))
            else:
                components.sort(key=len, reverse=True)
                # keep largest one only
                components.pop(0)
                for component in components:
                    super().remove_nodes_from(list(component))

        # clean up isolated node
        for node in nx.isolates(self):
            if node not in nodes_to_keep:
                super().remove_node(node)
