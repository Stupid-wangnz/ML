import numpy as np
import data_process
from lenet import *
from adam import *
from loss import *
import tqdm
import matplotlib.pyplot as plt


_loss = []
_acc = []

#超参数
epochs = 8
lr = 1e-3
batch_size = 256

def train_module(data, model):
    model.setlr(lr)
    best_acc = 0
    best_weight = None
    for e in range(epochs):
        for i in range(int(data['X_train'].shape[0]/batch_size)):
            X, y = data_process.get_batch(data["X_train"], data["y_train"], batch_size)
            y_pred = model.forward(X)
            loss, grad, acc = softmax_loss(y_pred, y)
            _loss.append(loss)
            _acc.append(acc) 
            model.backward(grad)

        val_X = data["X_val"]
        val_y = data["y_val"]
        y_pred = model.forward(val_X)
        y_pred = np.argmax(y_pred, axis=1)
        acc = np.mean(y_pred == val_y.reshape(1, val_y.shape[0]))
        if acc > best_acc:
            best_acc = acc
            best_weight = model.get_model_weight()

    plt.figure()
    plt.plot(_loss)
    plt.ylabel("softmax_loss")
    plt.savefig('loss.png')
    plt.figure()
    plt.plot(_acc)
    plt.ylabel("acc")
    plt.savefig('acc.png')

    return best_weight
