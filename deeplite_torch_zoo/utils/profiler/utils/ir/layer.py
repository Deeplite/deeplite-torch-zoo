__all__ = ['Layer']


class Layer:
    def __init__(self, name, inputs, outputs, weights=None, bias=None):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.weights = weights
        self.bias = bias

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name.lower()

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, inputs):
        self._inputs = inputs

    @property
    def outputs(self):
        return self._outputs

    @outputs.setter
    def outputs(self, outputs):
        self._outputs = outputs

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, bias):
        self._bias = bias

    def __repr__(self):
        text = "Node (name: {}, inputs: {}, outputs: {}, w: {}, b: {})".format(
                self.name, len(self.inputs), len(self.outputs), self.weights,
                self.bias)
        return text
