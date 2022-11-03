# coding=utf-8
from concurrent.futures import thread
from random import *
import numpy as np
import matplotlib.pyplot as plt

def softmax_regression(theta, x, y, iters, alpha):
    # TODO: Do the softmax regression by computing the gradient and 
    # the objective function value of every iteration and update the theta

    #the list of LOSS
    f = list()

    #惩罚项的力度
    lam = 0.0000025
    #个人习惯，可以不转置
    data=x.T
    #分批次训练

    batch_size = 200
    epoch=int(len(x)/batch_size)


    print(iters*epoch)

    for i in range(iters):
        for j in range(epoch):
            dataj=data[:,j*batch_size:(j+1)*batch_size]
            out=output(theta,dataj)
            #print(out.shape)
            y_hat=softmax(out)
            yj=y[:,j*batch_size:(j+1)*batch_size]
            #print(y_hat.shape)
            train_loss=loss(yj,y_hat) + lam*np.sum(theta**2)
            #print(train_loss,lam*np.sum(theta**2))
            #print(train_loss)
            if j==epoch-1:
                f.append(train_loss)

            #g is the gradient
            g=gradient(yj,y_hat,dataj) + lam*theta
            #g[:, 0] = g[:, 0] - lam * theta[:, 0]
            theta = theta - alpha * g

        #打乱数据,没啥用
        '''if i%50==0:
            index=[i for i in range (len(x))]
            np.random.shuffle(index)
            x=x[index]
            data=x.T
            y=(y.T[index]).T'''


    plt.title("Result Analysis")

    plt.plot(f, color='blue', label='loss')

    plt.legend() # 显示图例

    plt.xlabel("iteration times")

    plt.ylabel("loss")

    plt.savefig("test.png", dpi=600)

    plt.show()

    return theta

def output(w,x):
    return np.dot(w,x)

def softmax(output):
    output_exp=np.exp(output)
    exp_sum=output_exp.sum(axis=0)
    return (output_exp/exp_sum)

def loss(y,y_hat):
    batch_size=y.shape[1]
    y_hat=np.log(y_hat)
    l_sum=0.0
    for i in range(batch_size):
        l_sum+=np.dot(y_hat[:,i].T,y[:,i])
    return -(1.0/batch_size)*l_sum

def gradient(y,y_hat,data):
    batch_size=y.shape[1]
    return -(1.0/batch_size)*np.dot((y-y_hat),data.T)