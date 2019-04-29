#!/usr/local/bin/python
# INF 552 Assignment 4 Part 1
# Author: Zongdi Xu
# Mar 2019
import numpy as np
import time


def read_data(filename):
    input_f = open(filename, 'r')
    input_data = []
    for line in input_f.readlines():
        input_data.append([float(val) for val in line.split(',')])
    input_data = np.array(input_data)
    train_x = input_data[:, :-2]
    train_y = input_data[:, -2:-1]
    train_x = np.concatenate((train_x, np.ones((train_x.shape[0], 1))), axis=1)
    n, dimension = train_x.shape
    return n, dimension, train_x, train_y


pos_zero_threshold = 1e-7
neg_zero_threshold = -1e-7


def activate(val, threshold=neg_zero_threshold):
    activation_func = np.vectorize(lambda x: 1.0 if x > threshold else -1.0)
    return activation_func(val)


def predict(n, train_x, train_y, W):
    return activate(np.dot(train_x, W.reshape(-1, 1))*train_y)


def get_accuracy(n, hypothesis, train_y):
    return (np.abs(hypothesis-train_y) < pos_zero_threshold).sum().astype('float')/n


def perceptron(n, dimension, train_x, train_y, max_epoch, learning_rate):
    weight = np.array([-1.0]*dimension)

    for epoch in range(max_epoch):
        hypothesis = predict(n, train_x, train_y, weight)  # -train_y

        accuracy = get_accuracy(n, hypothesis, train_y)

        delta = hypothesis*learning_rate*train_x
        if (np.abs(hypothesis-train_y) < pos_zero_threshold).sum() == n:
            break
        weight -= np.squeeze(np.dot(delta.T, train_y).T)

    return weight, accuracy, hypothesis, epoch


if __name__ == '__main__':
    n, dimension, train_x, train_y = read_data('classification.txt')
    start_time = time.time()
    weight, accuracy, prediction, epoch = perceptron(
        n, dimension, train_x, train_y, max_epoch=5000, learning_rate=1e-6)
    print 'After %d epoch(s), %.3f s elapsed:' % (
        epoch, time.time()-start_time)
    print 'Weight matrix =', weight
    print 'Accuracy rate=%.2f' % accuracy
