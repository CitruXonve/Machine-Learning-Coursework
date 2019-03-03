#!/usr/local/bin/python
# INF 552 Assignment 1
# Name: Zongdi Xu
# USC ID: 5900-5757-70

import re
from sklearn import tree, preprocessing


def main():
    data_x, data_y, enc_x, enc_y = read_data('dt-data.txt')
    test_x = [['Moderate', 'Cheap', 'Loud', 'City-Center', 'No', 'No']]

    dtc = tree.DecisionTreeClassifier()
    dtc = dtc.fit(enc_x.transform(data_x).toarray(),
                  enc_y.transform(data_y).toarray())

    print 'The prediction test result of', test_x, 'is:', enc_y.inverse_transform(
        dtc.predict(enc_x.transform(test_x).toarray()))


def read_data(filename):
    input_f = open(filename, 'r')
    lines = input_f.readlines()

    predictor = [re.sub(r'^\s+|\s+$', '', word)
                 for word in list(filter(None, re.split(r'\(|\)|,|\n', lines[0])))]

    data = [list(filter(None, re.split(' |:|,|;|\n', rec)))[1:]
            for rec in lines[2:]]

    data_x = [line[:-1] for line in data]
    data_y = [line[-1:] for line in data]
    enc_x = preprocessing.OneHotEncoder()
    enc_x.fit(data_x)
    enc_y = preprocessing.OneHotEncoder()
    enc_y.fit(data_y)

    return data_x, data_y, enc_x, enc_y


if __name__ == '__main__':
    main()
