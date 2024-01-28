class Tensor:
    def __init__(self, name, dtype, shape=None, scope=None):
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.scope = scope

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        self._dtype = dtype.lower()

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape

    @property
    def scope(self):
        return self._scope

    @scope.setter
    def scope(self, scope):
        self._scope = scope.lower()

    def __repr__(self):
        text = "Tensor (name: {}, dtype: {}, shape: {})".format(
            self.name, self.dtype, self.shape
        )
        return text
