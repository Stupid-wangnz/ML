import numpy as np
from conv import *
from maxpool import *
from fullyconnect import *
from relu import *
from adam import *
from softmax import *

class LeNet:
    def __init__(self):
        self.c1 = Conv(1, 6, 5)
        self.relu1 = ReLu()
        self.s2 = MaxPool((2, 2), 2)
        self.c3 = Conv(6, 16, 5)
        self.relu2 = ReLu()
        self.s4 = MaxPool((2, 2), 2)
        self.c5 = FullyConnect(16 * 5 * 5, 120)
        self.relu3 = ReLu()
        self.fc6 = FullyConnect(120, 84)
        self.relu4 = ReLu()
        self.output = FullyConnect(84, 10)
        self.softmax = Softmax()
        self.adam = Adam([self.c1.W, self.c1.b, self.c3.W, self.c3.b, self.c5.W, self.c5.b, self.fc6.W, self.fc6.b, self.output.W, self.output.b])

    def forward(self, x):
        x = self.c1.forward(x)
        x = self.relu1.forward(x)
        x = self.s2.forward(x)
        x = self.c3.forward(x)
        x = self.relu2.forward(x)
        x = self.s4.forward(x)
        x = x.reshape(x.shape[0], -1)
        x = self.c5.forward(x)
        x = self.relu3.forward(x)
        x = self.fc6.forward(x)
        x = self.relu4.forward(x)
        x = self.output.forward(x)
        #x = self.softmax.forward(x)
        return x

    def backward(self, dy):
        params_grad = []

        #dy = self.softmax.backward(dy)
        
        dy, dW, db = self.output.backward(dy)
        params_grad.append(db)
        params_grad.append(dW)

        dy = self.relu4.backward(dy)
        dy, dW, db = self.fc6.backward(dy)
        params_grad.append(db)
        params_grad.append(dW)

        dy = self.relu3.backward(dy)
        dy, dW, db = self.c5.backward(dy)
        params_grad.append(db)
        params_grad.append(dW)

        dy = dy.reshape(dy.shape[0], 16, 5, 5)
        dy = self.s4.backward(dy)
        dy = self.relu2.backward(dy)
        dy, dW, db = self.c3.backward(dy)
        params_grad.append(db)
        params_grad.append(dW)

        dy = self.s2.backward(dy)
        dy = self.relu1.backward(dy)
        dy, dW, db = self.c1.backward(dy)
        params_grad.append(db)
        params_grad.append(dW)

        params_grad.reverse()
        self.adam.set_grad(params_grad) 
        self.adam.grad()
        #self.adam.sgd()

    def get_model_weight(self):
        return [self.c1.W, self.c1.b, self.c3.W, self.c3.b, self.c5.W, self.c5.b, self.fc6.W, self.fc6.b, self.output.W, self.output.b]

    def set_params(self, params):
        [self.c1.W, self.c1.b, self.c3.W, self.c3.b, self.c5.W, self.c5.b, self.fc6.W, self.fc6.b, self.output.W, self.output.b] = params
        self.c1.W = params[0]
        self.c1.b = params[1]
        self.c3.W = params[2]
        self.c3.b = params[3]
        self.c5.W = params[4]
        self.c5.b = params[5]
        self.fc6.W = params[6]
        self.fc6.b = params[7]
        self.output.W = params[8]
        self.output.b = params[9]
    
    def setlr(self, lr):
        self.adam.setlr(lr)