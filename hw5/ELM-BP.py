#!/usr/bin/python3
# -*- coding:utf8 -*-
import numpy as np
import re
import matplotlib.pyplot as plt


class SingeHiddenLayer(object):

    def __init__(self, X, y, num_hidden, num_out=1):
        self.data_x = np.atleast_2d(X)  # 判断输入训练集是否大于等于二维; 把x_train()取下来
        # a.flatten()把a放在一维数组中，不写参数默认是“C”，也就是先行后列的方式，也有“F”先列后行的方式； 把 y_train取下来
        self.data_y = np.array(y).flatten()
        self.num_data = len(self.data_x)  
        # shape[] 读取矩阵的长度，比如shape[0]就是读取矩阵第一维度的长度 (120行，4列，所以shape[0]==120,shapep[1]==4)
        self.num_feature = self.data_x.shape[1]
        self.num_hidden = num_hidden  # 隐藏层节点个数
        self.num_out = num_out

        # 随机生产权重（从-1，到1，生成（num_feature行,num_hidden列））
        self.w = np.random.uniform(-1, 1, (self.num_feature, self.num_hidden))

        self.out_w = np.random.uniform(-1, 1, (self.num_hidden, num_out))

        # 随机生成偏置，一个隐藏层节点对应一个偏置
        for i in range(self.num_hidden):
            b = np.random.uniform(-0.6, 0.6, (1, self.num_hidden))
            self.first_b = b

        # 生成偏置矩阵，以隐藏层节点个数4为行，样本数120为列
        for i in range(self.num_data - 1):
            b = np.row_stack((b, self.first_b))  # row_stack 以叠加行的方式填充数组
        self.b = b

        self.output_correction = np.zeros((self.num_hidden, self.num_out))
        self.input_correction = np.zeros((self.num_feature, self.num_hidden))

    # 定义sigmoid函数

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def back_propagate(self, x_train, y_train, learning_rate, max_iteration, correct):
        err_out = y_train- self.predict(x_train)
        print err_out

        error_hidden = np.array([0.0]*self.num_hidden)

        for h in range(self.num_hidden):
            error = 0.0
            for o in range(self.num_out):
                error += err_out[o] * self.out_w[h][o]
            try:
                error_hidden[h] = self.sigmoid_derivative(self.hidden[h]) * error
            except:
                print 'error,',self.hidden[h],self.sigmoid_derivative(self.hidden[h]),error
                break


        
        # update output weights
        for h in range(self.num_hidden):
            for o in range(self.num_out):
                change = err_out[o] * self.hidden[h]
                self.out_w[h][o] += learning_rate * change + correct * self.output_correction[h][o]
                self.output_correction[h][o] = change

        
        # update input weights
        for i in range(self.num_feature):
            for h in range(self.num_hidden):
                change = error_hidden[h] * self.input[i]
                self.w[i][h] += learn * change + correct * self.input_correction[i][h]
                self.input_correction[i][h] = change

        error = 0.0
        for o in range(len(y_train)):
            error += 0.5 * (y_train[o] - self.output[o]) ** 2
        return error

    def train(self, x_train, y_train, learning_rate=0.1, max_iteration=10, correct=0.0):
        # mul = np.dot(self.data_x, self.w)  # 输入乘以权重
        # print(self.data_x.shape)
        # add = mul + self.b  # 加偏置
        for _ in range(max_iteration):
            error = 0.0
            for i in range(len(x_train)):
                error+=self.back_propagate(x_train, y_train, learning_rate, max_iteration, correct)
            print _, error

    def predict(self, x_test):
        self.t_data = np.atleast_2d(x_test)  # 测试数据集
        self.num_tdata = len(self.t_data)  # 测试集的样本数
        self.pred_Y = np.zeros((x_test.shape[0]))  # 初始化

        self.input = x_test[:]

        print 'input=', self.t_data, self.input

        b = self.first_b

        # 扩充偏置矩阵，以隐藏层节点个数4为行，样本数30为列
        for i in range(self.num_tdata - 1):
            b = np.row_stack((b, self.first_b))  # 以叠加行的方式填充数组

        self.hidden = self.sigmoid(
            np.dot(self.t_data, self.w) + b)

        print 'hidden=',self.hidden

         # 预测
        self.output = np.dot(self.hidden, self.out_w)
        self.pred_Y = self.output[:]

        return(self.pred_Y)


def evaluate(predictions, targets, text):
    print(text)
    # 使用准确率方法验证
    # print(metrics.accuracy_score(y_true=y_test,y_pred=self.predy))
    print(rmse(predictions, targets))

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# stdsc = StandardScaler()   #StandardScaler类,利用接口在训练集上计算均值和标准差，以便于在后续的测试集上进行相同的缩放
# iris = load_iris()
# x,y = stdsc.fit_transform(iris.data),iris.target   #数据归一化

fig, ax = plt.subplots(1, 1)

def DrawLine(x, y, ax, tag=None, clr=None, lnstyle=None):
    if x is None:
        x = np.arange(y.size)
    index = x.argsort(0)
    ax.plot(x[index].reshape(x.size, -1), y[index].reshape(y.size, -1),
            label=tag, color=clr, linestyle=lnstyle)

def DrawDots(x, y, ax, tag=None, clr=None):
    if x is None:
        x = np.arange(y.size)
    index = x.argsort(0)
    ax.scatter(x[index].reshape(x.size, -1),
               y[index].reshape(y.size, -1), marker='^', color=clr)

def Work3(ax):
    x_train = np.array([
        [1, 2],
        [4, 4.5],
        [1, 0],
        [1, 1],
    ])
    y_train = np.array([[3], [9], [1], [2]])
    x_test=np.array([[1, 2],
        [4, 4.5],
        [1, 0],
        [1, 1],[1,.5],[2,2]])
    # y_test=y_train

    ELM = SingeHiddenLayer(x_train, y_train, 12)  # 训练数据集，训练集的label，隐藏层节点个数
    ELM.train(x_train, y_train)
    y_pred = ELM.predict(x_test)
    # evaluate(ELM.predict(x_train), y_train, '训练精度：')
    # evaluate(y_pred, y_test, '预测精度：')

    print(y_pred)
    # DrawDots(x_test, y_test, ax)
    # DrawDots(x_test, y_pred, ax, 'predicting curve', 'orange')

Work3(None)

def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))





def Work2():
    filename='./gestures/A/A_down_2.pgm'

    image = read_pgm(filename, byteorder='<')
#     print type(image), image
#     pyplot.imshow(image, pyplot.cm.gray)
#     pyplot.show()
    
    image = image.astype('float32')
    image /= np.max(image)
    plt.imshow(image, plt.cm.gray)
    plt.show()

    image = np.squeeze(image.reshape(1,-1))
    # print image.shape, image

    x_train = np.array([image])
    y_train = np.array([[1]])
    x_test = x_train

    ELM = SingeHiddenLayer(x_train, y_train, 12)  # 训练数据集，训练集的label，隐藏层节点个数
    ELM.train(x_train, y_train)
    y_pred = ELM.predict(x_test)
    # evaluate(ELM.predict(x_train), y_train, '训练精度：')
    # evaluate(y_pred, y_test, '预测精度：')

    print(y_pred)
    

# Work2()

# plt.show()