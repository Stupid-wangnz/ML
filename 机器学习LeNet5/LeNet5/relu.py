import numpy as np

class ReLu():
    def __init__(self):
        super().__init__()
        self.input = None
        self.output = None

    def forward(self, X):
        self.input = X.copy()
        self.output = np.maximum(0, X)
        return self.output

    def backward(self, back_grad):
        grad = self.input > 0
        return back_grad * grad