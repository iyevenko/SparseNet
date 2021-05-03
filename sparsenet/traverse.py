from sparsenet.ops import *

def forward_pass(order, feed_dict={}):
    """ Performs the forward pass, returning the output of the graph.
    Args:
        order: a topologically sorted array of nodes
        feed_dict: a dictionary values for placeholders.
    Returns:
        1. the final result of the forward pass.
        2. directly edits the graph to fill in its current values.
    """
    for node in order:

        if isinstance(node, Placeholder):
            node.value = feed_dict[node.name]

        elif isinstance(node, Operator):
            node.value = node.forward([prev_node.value for prev_node in node.inputs])

    return order[-1].value


def backward_pass(order, target_node=None):
    """ Perform the backward pass to retrieve gradients.
    Args:
        order: a topologically sorted array of graph nodes.
               by default, this assigns the gradient of the final node to 1
    Returns:
        gradients of nodes as listed in same order as input argument
    """
    seen = set()
    order[-1].gradient = 1
    for node in reversed(order):
        if isinstance(node, Operator):
            inputs = node.inputs
            grads = node.backward([x.value for x in inputs], dout=node.gradient)
            for inp, grad in zip(inputs, grads):
                if inp not in seen:
                    inp.gradient = grad
                else:
                    inp.gradient += grad
                seen.add(inp)
    return [node.gradient for node in order]


def apply_grads(order, grads, learning_rate=0.001):
    i = 0
    for node in order:
        if isinstance(node, Variable):
            node.value += -learning_rate*grads[i]

        i += 1
        node.gradient = None
