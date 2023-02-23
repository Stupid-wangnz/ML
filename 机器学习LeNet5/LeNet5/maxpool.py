import numpy as np

class MaxPool():
    def __init__(self, pool_size = [2,2],stride = 1):
        super().__init__()
        self.input = None
        self.output = None
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, X):
        self.input = X.copy()

        pool_h = self.pool_size[0]
        pool_w = self.pool_size[1]
        N, C, H, W = X.shape
        output_h = 1 + (H - pool_h) // self.stride
        output_w = 1 + (W - pool_w) // self.stride
        self.output = np.zeros((N, C, output_h, output_w))

        for h in range(output_h):
            for w in range(output_w):
                _x = h * self.stride
                _y = w * self.stride
                self.output[:, :, h, w] = np.max(X[:, :, _x:_x + pool_h, _y:_y + pool_w], axis=(2, 3))
        return self.output
        
    def backward(self, dy):
        pool_h = self.pool_size[0]
        pool_w = self.pool_size[1]
        N, C, H, W = self.input.shape
        output_h = 1 + (H - pool_h) // self.stride
        output_w = 1 + (W - pool_w) // self.stride
        dx = np.zeros_like(self.input)

        for h in range(output_h):
            for w in range(output_w):
                _x = h * self.stride
                _y = w * self.stride
                mask = np.zeros((N, C, pool_h * pool_w))
                mask[np.arange(N)[:, None], np.arange(C)[None, :], np.argmax(self.input[:, :, _x:_x + pool_h, _y:_y + pool_w].reshape((N, C, -1)), axis=2)] = 1
                dx[:, :, _x:_x + pool_h, _y:_y + pool_w] = mask.reshape((N, C, pool_h, pool_w)) * dy[:, :, h, w][:, :, None, None]
        
        return dx