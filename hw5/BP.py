#!/usr/local/bin/python
# -*- coding:utf8 -*-
import numpy as np
from sklearn.model_selection import train_test_split  # 数据集的分割函数
import matplotlib.pyplot as plt

# random.seed(0)
def rand(a, b):
    return (b - a) * np.random.random() + a
    
def __sigmoid(x):
    if x>=30.0: 
        return 1.0
    elif x<=-30.0:
        return 0.0
    return 1.0 / (1.0 + np.exp(-x))

def __sigmoid_derivative(x):
    return x * (1 - x)

def sigmoid(x):
    vfunc = np.vectorize(__sigmoid)
    return vfunc(x)

def sigmoid_derivative(x):
    vfunc = np.vectorize(__sigmoid_derivative)
    return vfunc(x)

class NeuralNetwork:
    def __init__(self, ni, nh, no):
        self.input_n = ni + 1
        self.hidden_n = nh
        self.output_n = no
        # init cells
        self.input_layer = np.ones((1,self.input_n))
        # init weights
        # random activate
        self.input_weights=np.random.uniform(-1,1,(self.input_n,self.hidden_n))
        self.output_weights=np.random.uniform(-1,1,(self.hidden_n,self.output_n))
        # init correction matrix
        self.input_correction = np.zeros((self.input_n, self.hidden_n))
        self.output_correction = np.zeros((self.hidden_n, self.output_n))

    def predict(self, x_train):
        # activate input layer
        for j in range(x_train.shape[0]):
            self.input_layer[:,j]=x_train[j]
        #输入层输出值
        # self.input_layer=x_train.copy()
        # activate hidden layer
        self.hidden_cells=sigmoid(np.dot(self.input_layer,self.input_weights))
        # activate output layer
        #输出层的激励函数是f(x)=x
        self.output_cells=np.round(sigmoid(np.dot(self.hidden_cells,self.output_weights)))
        return self.output_cells

    def back_propagate(self, x_train, y_train, learn, correct):#x,y,修改最大迭代次数， 学习率λ， 矫正率μ三个参数.
        # feed forward
        self.predict(x_train)
        # get output layer error
        #     # output_deltas[o] = sigmoid_derivative(self.output_cells[o]) * error
        output_deltas=y_train-self.output_cells

        # get hidden layer error
        hidden_deltas=np.dot(output_deltas,self.output_weights.T)*sigmoid_derivative(self.hidden_cells)

        # update output weights
        delta=np.dot(self.hidden_cells.T,output_deltas)
        self.output_weights+=learn*delta+correct*self.output_correction
        self.output_correction=delta

        # update input weights
        delta=np.dot(self.input_layer.T,hidden_deltas)
        self.input_weights+=learn*delta+correct*self.input_correction
        self.input_correction=delta

        # get global error
        # error=(y_train*self.output_cells)**2/len(y_train)
        # return np.sum(error)

    def train(self, x_train, y_train, limit=10000, learn=0.05, correct=0.1):
        # prior_error=0.0
        for j in range(limit):
            # error = 0.0
            for i in range(len(x_train)):
                # error += self.back_propagate(x_train[i], y_train[i], learn, correct)
                self.back_propagate(x_train[i], y_train[i], learn, correct)
            # if np.abs(prior_error-error)<epsilon:
            if np.sum(np.abs(y_train-self.test(x_train)))<1.0:
                print("Converge after " + str(j) + " epoch(s).")
                return
            # prior_error=error

        print "After " + str(j) + " epoch(s)."

    def test(self, x_test):
        y_pred = []
        for case in x_test:
            y_pred.append([np.squeeze(self.predict(case))])
        return np.array(y_pred)


if __name__ == '__main__':
    x_train = np.array([
        [1, 2],
        [4, 4.5],
        [1, 0],
        [1, 1],
    ])
    y_train = np.array([[1], [1], [0], [0]])
    nn=NeuralNetwork(x_train.shape[1], 12, y_train.shape[1])
    nn.train(x_train, y_train, learn=0.1, limit=1000)
    x_test = np.array([
        [1, 2],
        [4, 4.5],
        [1, 0],
        [1, 1],
        [1, 0.5],
        [2, 2]
    ])
    print nn.test(x_test).T
