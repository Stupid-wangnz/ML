import numpy as np
from train import *
import data_process
from lenet import *

data = data_process.get_mnist_data()
model = LeNet()

def accuracy(y, y_pred):
    return np.mean(y_pred == y.reshape(1, y.shape[0]))


def test(weight):
    X = data["X_test"]
    y = data["y_test"]
    model.set_params(weight)
    y_pred = model.forward(X)
    y_pred = np.argmax(y_pred, axis=1)
    acc = accuracy(y, y_pred)
    print(f"acc is : {acc}")

def main():
    best_weight = train_module(data, model)
    test(best_weight)

if __name__ == '__main__':
    main()
