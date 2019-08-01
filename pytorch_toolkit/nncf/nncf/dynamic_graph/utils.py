"""
 Copyright (c) 2019 Intel Corporation
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

import logging
from copy import deepcopy

from graphviz import Digraph
from torch import nn

from .. import dynamic_graph

logger = logging.getLogger(__name__)

graph_theme = {
    "background_color": "#FFFFFF",
    "fill_color": "#E8E8E8",
    "outline_color": "#000000",
    "font_color": "#000000",
    "font_name": "Times",
    "font_size": "10",
    "margin": "0,0",
    "padding": "1.0,1.0",
}


def to_networkx(context):
    import networkx as nx
    graph = nx.DiGraph()
    for node_name, node in context.graph.nodes.items():
        graph.add_node(node_name, type=node['type'], id=node['id'], scope='/'.join(node['scope']))
    for u, v in context.graph.edges:
        graph.add_edge(u, v)
    return graph


def dump_graph(context, path):
    import networkx as nx
    nx_graph = to_networkx(context)
    nx.drawing.nx_pydot.write_dot(nx_graph, path)


def draw_dot(context):
    graph = context.graph
    dot = Digraph()

    dot.attr("graph",
             bgcolor=graph_theme["background_color"],
             color=graph_theme["outline_color"],
             fontsize=graph_theme["font_size"],
             fontcolor=graph_theme["font_color"],
             fontname=graph_theme["font_name"],
             margin=graph_theme["margin"],
             # rankdir="LR",
             pad=graph_theme["padding"])
    dot.attr("node", shape="box",
             style="filled", margin="0,0",
             fillcolor=graph_theme["fill_color"],
             color=graph_theme["outline_color"],
             fontsize=graph_theme["font_size"],
             fontcolor=graph_theme["font_color"],
             fontname=graph_theme["font_name"])
    dot.attr("edge", style="solid",
             color=graph_theme["outline_color"],
             fontsize=graph_theme["font_size"],
             fontcolor=graph_theme["font_color"],
             fontname=graph_theme["font_name"])

    for node in graph.nodes:
        dot.node(graph.nodes[node]['name'])
        for child in graph.successors(node):
            dot.edge(node, child)
    return dot


def build_graph(module: nn.Module, args, kwargs, context_name, reset_context=False):
    logger.info("Building graph: {}".format(context_name))
    sd = deepcopy(module.state_dict())

    if reset_context:
        ctx = dynamic_graph.reset_context(context_name)
    else:
        ctx = dynamic_graph.get_context(context_name)
    with dynamic_graph.context(context_name):
        module(*args, **kwargs)

    module.load_state_dict(sd)
    return ctx
