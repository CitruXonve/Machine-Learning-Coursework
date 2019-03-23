import numpy as np


def read_data(filename):
    input_f = open(filename, 'r')
    input_data = []
    for line in input_f.readlines():
        input_data.append([float(val) for val in line.split(',')[:-1]])
    input_data = np.array(input_data)
    train_x = input_data[:, :-1]
    train_y = input_data[:, -1:]
    train_x = np.concatenate((train_x, np.ones((train_x.shape[0], 1))), axis=1)
    n, dimension = train_x.shape
    return n, dimension, train_x, train_y


def activation(val):
    return 1.0 if val > 0.0 else -1.0

def predict(n, train_x, train_y, W):
    output=[activation(np.sum(train_x[i,:]*W)*train_y[i,:]) for i in range(n)]
    return np.array(output).reshape(-1,1)


def perceptron(n, dimension, train_x, train_y, max_epoch, learning_rate):
    weight=np.zeros((1,dimension))

    for epoch in range(max_epoch):
        error=predict(n, train_x, train_y, weight)-train_y
        
        accuracy = 100.0-100.0*np.sum(np.abs(error))/n
        
        delta = error*learning_rate*train_x
        for i in range(n):
            weight-=np.array([delta[i,:]*train_y[i,:]])

        print 'Epoch #%d: accuracy_rate=%.2f%%'%(epoch+1,accuracy)

        if np.sum(error**2)<1.0:
            break
            
    return weight, accuracy


if __name__ == '__main__':
    n, dimension, train_x, train_y = read_data('classification.txt')
    weight, accuracy = perceptron(
        n, dimension, train_x, train_y, max_epoch=100, learning_rate=1e-3)
    print 'Weight matrix =', weight
    print 'Accuracy rate=%.2f%%' % accuracy
