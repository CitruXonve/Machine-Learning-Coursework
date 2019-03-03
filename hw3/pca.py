#!/usr/local/bin/python
# INF 552 Assignment 3
# By Zongdi Xu
# Date: Feb 28, 2019
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def main(filename):
    input_f = open(filename, 'r')
    data = input_f.readlines()
    x = []
    for record in data:
        x.append(record.split())
    x = np.array(x).astype(np.float)
    original_x = x.copy()

    # Get the mean of input data in every dimension
    num, dimension = x.shape
    num = x.shape[0]
    mean = np.sum(x, axis=0)/num

    # Step 1&2: get the covariance matrix
    cov_mat = (x - mean).T.dot((x - mean))/(num-1)

    # Step 3: calculate eigenvalues and eigenvectors
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i])
                 for i in range(len(eig_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort()
    eig_pairs.reverse()

    # Step 4: apply projection to input data points
    U = []
    target_dimension = 2
    for i in range(target_dimension):
        U.append(eig_pairs[i][1])
    U = np.array(U)

    result_x = np.dot(x, U.T)
    print result_x

    fig, ax = plt.subplots(1, 1)
    ax.scatter(result_x[:,0],result_x[:,1])
    plt.show()


if __name__ == '__main__':
    main('pca-data.txt')
