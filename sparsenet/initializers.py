from sparsenet.graph import *
from sparsenet.ops import relu, concat, add
import numpy as np


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
                    w_x = Variable(kernel[i, j], name=f'w_{layer},{i+1},{j+1}')
                    out.append(w_x*x)
                    j += 1
                b = Variable(bias[i], name=f'b_{layer},{i+1}')
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

            W = Variable(kernel, name=f'w_{i}')
            b = Variable(bias, name=f'b_{i}')

            x = x @ W + b
            if i < len(layer_widths) - 2:
                x = relu(x)

        self.tail = x[0]


class ConvolutionalGraph(Graph):

    def __init__(self, conv_widths, dense_widths, kernel_init, bias_init):
        super().__init__()
