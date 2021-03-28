from .graph import *
import numpy as np

# Adds an arbitrary amount of numpy array inputs
class add(Operator):
    count = 0

    def __init__(self, *inputs, name=None):
        super().__init__(name)
        self.inputs = inputs
        self.name = f'add/{add.count}' if name is None else name
        add.count += 1

    def forward(self, inputs):
        return sum(inputs)

    def backward(self, inputs, dout):
        # reduce grads by sum if inputs were broadcasted
        return [np.full_like(i, dout) if np.ndim(i) >= np.ndim(dout) else
                np.sum(dout, tuple(np.arange(np.ndim(dout)-np.ndim(i))))
                for i in inputs]

# Sum of a single numpy array
class reduce_sum(Operator):
    count = 0

    def __init__(self, *inputs, name=None):
        super().__init__(name)
        self.inputs = inputs
        self.name = f'sum/{reduce_sum.count}' if name is None else name
        reduce_mean.count += 1

    def forward(self, inputs):
        return np.sum(inputs[0])

    def backward(self, inputs, dout):
        return [np.full_like(inputs[0], dout)]

# Mean of a single numpy array
class reduce_mean(Operator):
    count = 0

    def __init__(self, *inputs, name=None):
        super().__init__(name)
        self.inputs = inputs
        self.name = f'mean/{reduce_mean.count}' if name is None else name
        reduce_mean.count += 1

    def forward(self, inputs):
        return np.mean(inputs[0])

    def backward(self, inputs, dout):
        return [np.full_like(inputs[0], dout) / np.size(inputs[0])]

# Multiply a numpy array by a constant or another numpy array
class multiply(Operator):
    count = 0

    def __init__(self, a, b, name=None):
        super().__init__(name)
        self.inputs = [a, b]
        self.name = f'mul/{multiply.count}' if name is None else name
        multiply.count += 1

    def forward(self, inputs):
        a, b = inputs
        return a * b

    def backward(self, inputs, dout):
        a, b = inputs
        da, db = dout * b, dout * a
        return np.sum(da, tuple(np.arange(np.ndim(da)-np.ndim(a)))), \
               np.sum(db, tuple(np.arange(np.ndim(db)-np.ndim(b))))

# Divide a numpy array by a constant or another numpy array
class divide(Operator):
    count = 0

    def __init__(self, a, b, name=None):
        super().__init__(name)
        self.inputs = [a, b]
        self.name = f'div/{divide.count}' if name is None else name
        divide.count += 1

    def forward(self, inputs):
        a, b = inputs
        return a/b

    def backward(self, inputs, dout):
        a, b = inputs
        return dout / b, dout * a / np.power(b, 2)

# Take a numpy array to the power of a constant or vice versa
class power(Operator):
    count = 0

    def __init__(self, a, b, name=None):
        super().__init__(name)
        self.inputs = [a, b]
        self.name = f'pow/{power.count}' if name is None else name
        power.count += 1

    def forward(self, inputs):
        a, b = inputs
        return np.power(a, b)

    def backward(self, inputs, dout):
        a, b = inputs
        return dout * b * np.power(a, (b - 1)), dout * np.log(a) * np.power(a, b)

# Take the log of a numpy array or constant
class log(Operator):
    count = 0

    def __init__(self, a, name=None):
        super().__init__(name)
        self.inputs = [a]
        self.name = f'log/{log.count}' if name is None else name
        log.count += 1

    def forward(self, inputs):
        return np.log(inputs[0])

    def backward(self, inputs, dout):
        return [dout / inputs[0]]

# Get the element at the ith index in a numpy array
class array_slice(Operator):
    count = 0

    def __init__(self, a, i, name=None):
        super().__init__(name)
        self.inputs = [a, i]
        self.name = f'slice/{array_slice.count}' if name is None else name
        array_slice.count += 1

    def forward(self, inputs):
        a, i = inputs
        return a[i]

    def backward(self, inputs, dout):
        a, i = inputs
        grad = np.zeros_like(a)
        grad[i] = dout
        return grad, 0

# Matrix multiply two numpy arrays
class matmul(Operator):
    count = 0

    def __init__(self, a, b, name=None):
        super().__init__(name)
        self.inputs = [a, b]
        self.name = f'matmul/{matmul.count}' if name is None else name
        matmul.count += 1

    def forward(self, inputs):
        a, b = inputs
        return a @ b

    def backward(self, inputs, dout):
        a, b = inputs
        return dout @ b.T, a.T @ dout

# Concatenate a list of numpy arrays (used in output of most models)
class concat(Operator):
    count = 0

    def __init__(self, inputs, name=None):
        super().__init__()
        self.inputs = inputs
        self.name = f'concat/{concat.count}' if name is None else name
        concat.count += 1

    def forward(self, inputs):
        return np.concatenate(inputs)

    def backward(self, inputs, dout):
        return np.split(dout, len(inputs))

# Element-wise maximum of a numpy array
class maximum(Operator):
    count = 0

    def __init__(self, a, name=None):
        super().__init__()
        self.inputs = [a]
        self.name = f'max/{maximum.count}' if name is None else name
        maximum.count += 1

    def forward(self, inputs):
        return np.max(inputs[0])

    def backward(self, inputs, dout):
        return [dout * np.greater_equal(inputs[0], self.value)]

# Element-wise absolute value of a numpy array
class absolute_value(Operator):
    count = 0

    def __init__(self, a, name=None):
        super().__init__()
        self.inputs = [a]
        self.name = f'abs/{absolute_value.count}' if name is None else name
        absolute_value.count += 1

    def forward(self, inputs):
        return np.abs(inputs[0])

    def backward(self, inputs, dout):
        return [dout * np.sign(inputs[0])]

# Element wise maximum with 0 of a numpy array
class relu(Operator):
    count = 0

    def __init__(self, a, name=None):
        super().__init__()
        self.inputs = [a]
        self.name = f'relu/{relu.count}' if name is None else name
        relu.count += 1

    def forward(self, inputs):
        return np.maximum(inputs[0], 0)

    def backward(self, inputs, dout):
        return [dout * np.greater(inputs[0], 0)]


def mean_absolute_error(label, y_pred):
    return reduce_mean(absolute_value(label - y_pred))

def softmax(logits):
    exps = Constant(np.e, name='e') ** (logits-maximum(logits))
    return exps / reduce_sum(exps)

def sparse_categorical_cross_entropy(label, y_pred):
    # label -> int [0, 9]
    # y_pred -> numpy array of logits
    probs = softmax(y_pred)
    return reduce_mean(-log(probs[label]))

def node_wrapper(func, self, other):
    """ Check to make sure that the two things we're comparing are
    actually graph nodes. Also, if we use a constant, automatically
    make a Constant node for it"""
    if isinstance(other, Node):
        return func(self, other)
    if isinstance(other, float) or isinstance(other, int):
        return func(self, Constant(other))
    raise TypeError("Incompatible types.")

Node.__add__ = lambda self, other: node_wrapper(add, self, other)
Node.__sub__ = lambda self, other: node_wrapper(add, self, -other)
Node.__mul__ = lambda self, other: node_wrapper(multiply, self, other)
Node.__truediv__ = lambda self, other: node_wrapper(divide, self, other)
Node.__neg__ = lambda self: node_wrapper(multiply, self, Constant(-1.0))
Node.__pow__ = lambda self, other: node_wrapper(power, self, other)
Node.__matmul__ = lambda self, other: node_wrapper(matmul, self, other)
Node.__getitem__ = lambda self, index: node_wrapper(array_slice, self, index)
