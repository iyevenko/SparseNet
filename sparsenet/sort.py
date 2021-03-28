from sparsenet.ops import *

from graphviz import Digraph


def topological_sort(graph, head_node=None):
    """Performs topological sort of all nodes prior to and
    including the head_node.
    Args:
        graph: the computational graph. This is the global value by default
        head_node: last node in the forward pass. The "result" of the graph.
    Returns:
        a sorted array of graph nodes.
    """
    vis = set()
    ordering = []

    def _dfs(node):
        if node not in vis:
            vis.add(node)
            if isinstance(node, Operator):
                for input_node in node.inputs:
                    _dfs(input_node)
            ordering.append(node)

    if head_node is None:
        for node in graph.operators:
            _dfs(node)
    else:
        _dfs(head_node)

    return ordering



def plot(graph):
    f = Digraph()
    f.attr(rankdir='LR', size='10, 8')
    f.attr('node', shape='circle')

    for node in graph:
        shape = 'box' if isinstance(node, Placeholder) else 'circle'
        f.node(node.name, label=node.name.split('/')[0], shape=shape)
    for node in graph:
        if isinstance(node, Operator):
            for e in node.inputs:
                f.edge(e.name, node.name, label=e.name)
    return f