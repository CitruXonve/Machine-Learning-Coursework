#!/usr/local/bin/python
# INF 552 Assignment 4 Part 3
# Author: Zongdi Xu
# Mar 2019

import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import datasets


def read_data(filename):
    input_f = open(filename, 'r')
    input_data = []
    for line in input_f.readlines():
        input_data.append([float(val) for val in line.split(',')])
    input_data = np.array(input_data)
    train_x = input_data[:, :-2]
    train_y = input_data[:, -1]
    train_x = np.concatenate((train_x, np.ones((train_x.shape[0], 1))), axis=1)
    n, dimension = train_x.shape
    return n, dimension, train_x, train_y.reshape(-1, 1)


class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=10000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def sigmoid(self, val):
        return 1 / (1 + np.exp(-val))

    def fit(self, X, y):

        # weights initialization
        self.theta = np.zeros(X.shape[1])

        # gradient descent
        for i in range(self.max_iter):
            hypothesis = self.sigmoid(np.dot(X, self.theta))
            gradient = np.dot(X.T, (hypothesis - y)) / y.size
            self.theta -= self.learning_rate * gradient

        return i

    def predict(self, X):
        return self.sigmoid(np.dot(X, self.theta)).round()


if __name__ == '__main__':
    n, dimension, X, y = read_data('classification.txt')
    y = np.squeeze(y)
    y = (y+1.1).astype('int')/2

    model = LogisticRegression(0.1, 70000)

    start_time = time.time()
    epoch = model.fit(X, y)
    print 'After %d epoch(s), %.3f s elapsed:' % (
        epoch, time.time()-start_time)

    prediction = model.predict(X).astype('int')

    print 'Weight matrix=', model.theta
    print 'accuracy_rate=', (prediction == y).sum()*1.0/len(prediction)
