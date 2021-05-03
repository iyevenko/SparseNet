import math

from sparsenet.graph import *
from sparsenet.ops import *
import numpy as np
from scipy.sparse import csr_matrix


def tensor_to_kernel(t, prev):
    # https://stackoverflow.com/questions/16798888/2-d-convolution-as-a-matrix-matrix-multiplication

    filters = []

    for k in t:
        m = k.shape[0]
        n = int(math.sqrt(prev.shape[0]))
        assert n**2 == prev.shape[0]

        N = prev.shape[0]
        M = (n-m+1) ** 2

        shifts = np.array([i for i in range(n*(n-m+1)) if i % n < m])
        padded_filter = np.pad(k, ((0, n-m),(0, n-m,)))
        unrolled_filter = padded_filter.flatten()[np.newaxis,:]
        filter = np.repeat(unrolled_filter, M, axis=0)

        # funky rolling of each row
        rows, column_indices = np.ogrid[:filter.shape[0], :filter.shape[1]]
        column_indices = column_indices - shifts[:, np.newaxis]
        filter = filter[rows, column_indices]

        filters.append(filter)

    return filters



class Initializer():

    def __init__(self, func):
        self.func = func

    def get(self, shape):
        return self.func(shape)


class ConstantInitializer(Initializer):

    def __init__(self, value):
        self.value = float(value)

        def _func(shape):
            return np.full(shape, self.value)

        super().__init__(_func)


class NormalInitializer(Initializer):

    def __init__(self, scale):
        self.scale = scale

        def _func(shape):
            return np.random.normal(size=shape, scale=self.scale)

        super().__init__(_func)

class ConstantKernelInitializer(Initializer):

    def __init__(self, value):
        self.value = float(value)

        def _func(filter):
            width, num, prev_dim = filter
            shape = (num, width, width)

            tensor = np.full(shape, self.value)
            kernel = tensor_to_kernel(tensor, prev_dim)
            kernel = np.array(csr_matrix(kernel[i]) for i in range(kernel.shape[0]))
            return kernel

        super().__init__(_func)



class BasicDenseGraph(Graph):

    def __init__(self, layer_widths, kernel_init, bias_init):
        super().__init__()
        np.random.seed(0)
        self.inputs = [Placeholder(name=f'in/{i}') for i in range(layer_widths[0])]

        prev_layer = self.inputs
        layer = 1

        for width in layer_widths[1:]:
            # kernel_init = np.random.normal(scale=0.05, size=(width, layer_widths[layer-1]))
            kernel = kernel_init.get((width, layer_widths[layer-1]))
            bias = bias_init.get((width))
            curr_layer=[]
            for i in range(width):
                j = 0
                out = []
                for x in prev_layer:
                    w_x = Variable(kernel[i, j], name=f'w/{layer},{i+1},{j+1}')
                    out.append(w_x*x)
                    j += 1
                b = Variable(bias[i], name=f'b/{layer},{i+1}')
                out.append(b)
                out = add(*out)
                if layer < len(layer_widths)-1:
                    out = relu(out)

                curr_layer.append(out)
            prev_layer = curr_layer
            layer +=1

        self.tail = concat(prev_layer)


class VectorizedDenseGraph(Graph):

    def __init__(self, layer_widths, kernel_init, bias_init):
        super().__init__()
        self.inputs = Placeholder(name='in/0')

        x = self.inputs
        for i in range(len(layer_widths)-1):
            kernel = kernel_init.get((layer_widths[i], layer_widths[i+1]))
            bias = bias_init.get((layer_widths[i+1]))

            W = Variable(kernel, name=f'w/{i}')
            b = Variable(bias, name=f'b/{i}')

            x = x @ W + b
            if i < len(layer_widths) - 2:
                x = relu(x, name=f'a/{i}')
            # else:
            #     self.operators['out'] = self.operators[f'add/{i}']
            #     del self.operators[f'add/{i}']
            #     self.operators['out'].name = 'out'

        self.tail = array_slice(x, Constant(0), name='out')
        self.tail.name = 'out'


class ConvolutionalGraph(Graph):

    def __init__(self, filters, dense_widths, filter_init, bias_init):
        # Filters must be a list of (filter_size, num_filters)

        super().__init__()
        self.inputs = Placeholder(name='in/0')

        x = self.inputs
        for i in range(len(filters)):
            filter = filter_init.get(filters[i])
            bias = bias_init.get()

            f = Variable(filter, name=f'f_{i}')
            b = Variable(bias, name=f'b_{i}')

            x = pad(x, ())
            x = relu(f @ x + b)



