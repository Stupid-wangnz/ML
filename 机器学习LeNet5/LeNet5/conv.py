import numpy as np

class Conv():
    def __init__(self, input_channels, kernels, filter_size = 5, stride = 1, padding = 0):
        super().__init__()
        self.input = None
        self.output = None
        self.input_channels = input_channels
        self.kernels = kernels

        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding

        self.W = np.random.normal(scale=1e-3, size=(kernels, input_channels, filter_size, filter_size))
        self.b = np.zeros(kernels)

    def forward(self, X):
        self.input = X.copy()

        N, C, H, W = X.shape
        padded = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        output_h = 1 + (H + 2 * self.padding - self.filter_size) // self.stride
        output_w = 1 + (W + 2 * self.padding - self.filter_size) // self.stride
        self.output = np.zeros((N, self.kernels, output_h, output_w))

        for h in range(output_h):
            for w in range(output_w):
                _x = h * self.stride
                _y = w * self.stride
                temp_x = padded[:, :, _x:_x + self.filter_size, _y:_y + self.filter_size].reshape((N, 1, self.input_channels, self.filter_size, self.filter_size))
                temp_w = self.W.reshape((1, self.kernels, self.input_channels, self.filter_size, self.filter_size))
                self.output[:, :, h, w] = np.sum(temp_x * temp_w, axis=(2, 3, 4)) + self.b
        return self.output

    def backward(self, dy):
        N, C, H, W = self.input.shape
        padded = np.pad(self.input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        output_h = 1 + (H + 2 * self.padding - self.filter_size) // self.stride
        output_w = 1 + (W + 2 * self.padding - self.filter_size) // self.stride
        dW = np.zeros((self.kernels, self.input_channels, self.filter_size, self.filter_size))
        db = np.zeros(self.kernels)
        dx = np.zeros_like(padded)
        for h in range(output_h):
            for w in range(output_w):
                _x = h * self.stride
                _y = w * self.stride
                
                temp_grad = dy[:, :, h, w].reshape((N, 1, 1, 1, self.kernels))
                temp_w = self.W.transpose((1, 2, 3, 0)).reshape((1, self.input_channels, self.filter_size, self.filter_size, self.kernels))
                #dx
                dx[:, :, _x:_x + self.filter_size, _y:_y + self.filter_size] += np.sum(temp_grad * temp_w, axis=4)
                #dw = dx * x
                temp_grad = dy[:, :, h, w].T.reshape((self.kernels, 1, 1, 1, N))
                x_T = padded[:, :, h * self.stride:h * self.stride + self.filter_size, w * self.stride:w * self.stride + self.filter_size].transpose((1, 2, 3, 0))
                dW += np.sum(temp_grad * x_T, axis=4)
                #db
                db += np.sum(dy[:, :, h, w], axis=0)
        dx = dx[:, :, self.padding:self.padding + H, self.padding:self.padding + W]
        return dx, dW, db