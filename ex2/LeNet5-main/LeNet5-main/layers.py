import numpy as np


class Layer(object):
    def __init__(self):
        pass

    def forward(self, X):
        pass


    def backward(self, back_grad):
        pass

#Convolutions
class Conv(Layer):
    def __init__(self, input_channels, kernels, filter_size = 5, stride = 1, padding = 0):
        super().__init__()
        self.input = None
        self.output = None
        self.input_channels = input_channels
        self.kernels = kernels

        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding

        self.W = {'value': np.random.normal(scale=1e-3, size=(kernels, input_channels, filter_size, filter_size)),
                  'grad': np.zeros((kernels, input_channels, filter_size, filter_size))}
        self.b = {'value': np.zeros(kernels),
                  'grad': np.zeros(kernels)}

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
                temp_w = self.W['value'].reshape((1, self.kernels, self.input_channels, self.filter_size, self.filter_size))
                self.output[:, :, h, w] = np.sum(temp_x * temp_w, axis=(2, 3, 4)) + self.b['value']
        return self.output

    def backward(self, back_grad):
        N, C, H, W = self.input.shape
        padded = np.pad(self.input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        output_h = 1 + (H + 2 * self.padding - self.filter_size) // self.stride
        output_w = 1 + (W + 2 * self.padding - self.filter_size) // self.stride
        self.W['grad'] = np.zeros((self.kernels, self.input_channels, self.filter_size, self.filter_size))
        self.b['grad'] = np.zeros(self.kernels)
        grad = np.zeros_like(padded)
        for h in range(output_h):
            for w in range(output_w):
                _x = h * self.stride
                _y = w * self.stride
                
                temp_grad = back_grad[:, :, h, w].reshape((N, 1, 1, 1, self.kernels))
                temp_w = self.W['value'].transpose((1, 2, 3, 0)).reshape((1, self.input_channels, self.filter_size, self.filter_size, self.kernels))
                #dx
                grad[:, :, _x:_x + self.filter_size, _y:_y + self.filter_size] += np.sum(temp_grad * temp_w, axis=4)
                #dw = grad * x
                temp_grad = back_grad[:, :, h, w].T.reshape((self.kernels, 1, 1, 1, N))
                x_T = padded[:, :, h * self.stride:h * self.stride + self.filter_size, w * self.stride:w * self.stride + self.filter_size].transpose((1, 2, 3, 0))
                self.W['grad'] += np.sum(temp_grad * x_T, axis=4)
                #db
                self.b['grad'] += np.sum(back_grad[:, :, h, w], axis=0)
        grad = grad[:, :, self.padding:self.padding + H, self.padding:self.padding + W]
        return grad

class ReLu(Layer):
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


class MaxPool(Layer):
    def __init__(self, pool_size=[2,2],stride=1):
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
        
    def backward(self, back_grad):
        pool_h = self.pool_size[0]
        pool_w = self.pool_size[1]
        N, C, H, W = self.input.shape
        output_h = 1 + (H - pool_h) // self.stride
        output_w = 1 + (W - pool_w) // self.stride
        grad = np.zeros_like(self.input)

        for h in range(output_h):
            for w in range(output_w):
                _x = h * self.stride
                _y = w * self.stride
                mask = np.zeros((N, C, pool_h * pool_w))
                mask[np.arange(N)[:, None], np.arange(C)[None, :], np.argmax(self.input[:, :, _x:_x + pool_h, _y:_y + pool_w].reshape((N, C, -1)), axis=2)] = 1
                grad[:, :, _x:_x + pool_h, _y:_y + pool_w] = mask.reshape((N, C, pool_h, pool_w)) * back_grad[:, :, h, w][:, :, None, None]
        
        return grad

class FullyConnect(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input = None
        self.output = None
        self.input_size = input_size
        self.output_size = output_size
        self.W = {'value': np.random.normal(scale=1e-3, size=(input_size, output_size)),
                  'grad': np.zeros((input_size, output_size))}
        self.b = {'value': np.zeros(output_size),
                  'grad': np.zeros(output_size)}

    def forward(self, X):
        self.input = X.copy()
        self.output = np.dot(X, self.W['value']) + self.b['value']
        return self.output

    def backward(self, back_grad):
        grad = np.dot(back_grad, self.W['value'].T)
        self.W['grad'] = np.dot(self.input.T, back_grad)
        self.b['grad'] = np.sum(back_grad, axis=0)
        return grad


