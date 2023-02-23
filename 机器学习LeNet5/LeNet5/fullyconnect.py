import numpy as np

class FullyConnect():
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input = None
        self.output = None
        self.input_size = input_size
        self.output_size = output_size
        self.W = np.random.normal(scale=1e-3, size=(input_size, output_size))
        self.b = np.zeros(output_size)

    def forward(self, X):
        self.input = X.copy()
        self.output = np.dot(X, self.W) + self.b
        return self.output

    def backward(self, dy):
        dx = np.dot(dy, self.W.T)
        dW = np.dot(self.input.T, dy)
        db = np.sum(dy, axis=0)
        return dx, dW, db