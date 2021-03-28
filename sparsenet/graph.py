class Graph():

    def __init__(self):
        self.operators = set()
        self.constants = set()
        self.variables = set()
        self.placeholders = set()
        global _g
        _g = self

    def apply_gradients(self, learning_rate):
        for var in self.variables:
            var.value += -learning_rate*var.gradient
            var.gradient = None
        for op in self.operators:
            op.gradient = None
        for c in self.constants:
            c.gradient = None
        for p in self.placeholders:
            p.gradient = None

    def reset_counts(self, root):
        if hasattr(root, 'count'):
            root.count = 0
        else:
            for child in root.__subclasses__():
                self.reset_counts(child)

    def reset_session(self):
        try:
            del _g
        except:
            pass
        self.reset_counts(Node)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.reset_session()


### This won't do anything other than allow us to check
### if in object is a Graph node or not
class Node:
    def __init__(self):
        pass


class Placeholder(Node):
    """An placeholder node in the computational graph. This holds
    a node, and awaits further input at computation time.
    Args:
        name: defaults to "Plc/"+count
    """
    count = 0

    def __init__(self, name=None):
        super().__init__()
        _g.placeholders.add(self)
        self.value = None
        self.gradient = None
        self.name = f"Plc/{Placeholder.count}" if name is None else name
        Placeholder.count += 1

    def __repr__(self):
        return f"Placeholder: name:{self.name}, value:{self.value}"


class Constant(Node):
    """A constant node in the computational graph.
    Args:
        value: a property protected value that prevents user
               from reassigning value
        name: defaults to "Const/"+count
    """
    count = 0

    def __init__(self, value, name=None):
        super().__init__()
        _g.constants.add(self)
        self.value = value
        self.gradient = None
        self.name = f"Const/{Constant.count}" if name is None else name
        Constant.count += 1

    def __repr__(self):
        return f"Constant: name:{self.name}, value:{self.value}"



class Variable(Node):
    """An variable node in the computational graph. Variables are
    automatically tracked during graph computation.
    Args:
        value: a mutable value
        name: defaults to "Var/"+count
    """
    count = 0

    def __init__(self, value, name=None):
        super().__init__()
        _g.variables.add(self)
        self.value = value
        self.gradient = None
        self.name = f"Var/{Variable.count}" if name is None else name
        Variable.count += 1

    def __repr__(self):
        return f"Variable: name:{self.name}, value:{self.value}"


class Operator(Node):
    """An operator node in the computational graph.
    Args:
        name: defaults to "operator name/"+count
    """

    def __init__(self, name='Op'):
        super().__init__()
        _g.operators.add(self)
        self.value = None
        self.inputs = []
        self.gradient = None
        self.name = name

    def __repr__(self):
        return f"Operator: name:{self.name}"

    def forward(self, inputs):
        pass

    def backward(self, inputs, dout):
        pass
