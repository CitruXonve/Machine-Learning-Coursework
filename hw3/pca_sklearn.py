#!/usr/local/bin/python
# INF 552 Assignment 3
# By Zongdi Xu
# Date: Feb 28, 2019
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def main(filename):
    input_f = open(filename, 'r')
    data = input_f.readlines()
    x = []
    for record in data:
        x.append(record.split())
    x = np.array(x).astype(np.float)
    original_x = x.copy()

    n, dimension = x.shape
    pca = PCA(n_components=2)
    result_x = pca.fit_transform(x)
    print result_x

    fig, ax = plt.subplots(1, 1)
    ax.scatter(result_x[:,0],result_x[:,1])
    plt.show()


if __name__ == '__main__':
    main('pca-data.txt')
