from sparsenet.sort import *
from sparsenet.traverse import *
from sparsenet.initializers import *

import tensorflow as tf

#
# with Graph() as g:
#     out = sparse_categorical_cross_entropy(Placeholder(name='label'), Placeholder(name='y_pred'))
#     order = topological_sort(g, out)
#     plot = plot(order)
#     plot.render(view=True)

# #
#
# with VectorizedDenseGraph([4, 4], NormalInitializer(0.05), NormalInitializer(0.05)) as g:
#     order = topological_sort(g, g.tail)
#     print([(v.name, v.value.shape) for v in sorted(g.variables, key=lambda x: x.name)])
#     plot = plot(order)
#     plot.render(view=True)

x = np.arange(9)
t = np.arange(4).reshape((1, 2, 2))
k = tensor_to_kernel(t, x)[0]

print(k @ x)

# val1, val2, val3 = 0.9, 0.4, -1.3

# with Graph() as g:
#     x = Variable(val1, name='x')
#     y = Variable(val2, name='y')
#     c = Constant(val3, name='c')
#     z = c*x+relu(y*x+c)
#
#     order = topological_sort(g, z)
#     res = forward_pass(order)
#     grads = backward_pass(order)
#
#     print("Node ordering:")
#     for node in order:
#         print(node)
#
#     print('-'*10)
#     # print(f"Forward pass expected: {(val1*val2+val3)*val3+val1}")
#     print(f"Forward pass computed: {res}")
#
#     dzdx_node = [a for a in order if a.name == 'x'][0]
#     dzdy_node = [a for a in order if a.name == 'y'][0]
#     dzdc_node = [a for a in order if a.name == 'c'][0]
#
#     # print(f"dz/dx expected = {val3 * val2 + 1}")
#     print(f"dz/dx computed = {dzdx_node.gradient}")
#     # print(f"dz/dy expected = {val1 * val3}")
#     print(f"dz/dy computed = {dzdy_node.gradient}")
#     # print(f"dz/dc expected = {val1 * val2 + 2 * val3}")
#     print(f"dz/dc computed = {dzdc_node.gradient}")
#
#     plot = plot(order)
#     plot.render(view=True)