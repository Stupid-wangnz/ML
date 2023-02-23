import numpy as np

class Adam:
    def __init__(self, params, alpha=1e-3, beta1=0.9, beta2=0.999, epsilon = 1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.iter = 0
        self.alpha = alpha
        #参数
        self.params = params
        #梯度
        self.params_grad = None
        self.m = []
        self.theta = []
        self.v = None
        for i in range(len(self.params)):
            self.m.append(np.zeros_like(self.params[i]))

    def setlr(self, alpha):
        self.alpha = alpha

    def set_grad(self, grad):
        self.params_grad = grad
        if self.v is None :
            self.v = []
            for i in range(len(self.params)):
                self.v.append(np.zeros_like(self.params_grad[i]))

    def grad(self):

        self.iter += 1
        for i in range(len(self.params_grad)):
            self.m[i] = (1 - self.beta1) * (self.params_grad[i]) + self.beta1 * self.m[i]
            self.v[i] = (1 - self.beta2) * (self.params_grad[i] ** 2) + self.beta2 * self.v[i]
            m_c = self.m[i] / (1.0 - self.beta1 ** self.iter)
            v_c = self.v[i] / (1.0 - self.beta2 ** self.iter)
            self.params[i] -= (self.alpha * m_c) / (np.sqrt(v_c) + self.epsilon)

    def sgd(self):
        for i in range(len(self.params_grad)):
            self.params[i] -= self.alpha*self.params_grad[i]