# -*- coding: utf-8 -*-
# INF 552 Assignment 2
# Name: Zongdi Xu, Wenkai Xu
# USC ID: 5900-5757-70
# Python version 2.7
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
import math
from random import randint
from numpy.random import rand
from numpy.linalg import norm


def randomCenteroid(clusterNum, f):
    x_min = min(f[:, 0])
    x_max = max(f[:, 0])
    y_min = min(f[:, 1])
    y_max = max(f[:, 1])
    p = rand(clusterNum, 2)
    p = p * np.array([np.ones(clusterNum) * (x_max - x_min), np.ones(clusterNum) * (y_max - y_min)]
                     ).transpose() + np.array([np.ones(clusterNum) * x_min, np.ones(clusterNum) * y_min]).transpose()
    return p


def InitAllPoint():
    res = []
    f = open('clusters.txt', 'r')
    for line in f.readlines():
        res.append([float(x) for x in line.split(',')])

    res = np.array(res)
    x_min = min(res[:, 0])
    x_max = max(res[:, 0])
    y_min = min(res[:, 1])
    y_max = max(res[:, 1])
    return res, x_min, x_max, y_min, y_max


def multivariate_gaussian(pos, mu, Sigma):
    dimension = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / np.sqrt((2*np.pi)**dimension * Sigma_det)


def draw_cluster(ax, x_min, x_max, y_min, y_max, clusterNum, phi, mu, sigma):
    x = np.arange(x_min - 1, x_max + 1, 0.1)
    y = np.arange(y_min - 1, y_max + 1, 0.1)
    x, y = np.meshgrid(x, y)
    for j in range(clusterNum):
        z = multivariate_gaussian(
            np.array([x.ravel(), y.ravel()]).T, mu[j], sigma[j])
        z = np.array([z])
        z = z.reshape(len(np.arange(y_min - 1, y_max+1, 0.1)), -1)
        ax.contour(x, y, z, colors='orange', alpha=phi[j])


def Gaussian(clusterNum, Maximum_iteration, deviation):
    node, x_min, x_max, y_min, y_max = InitAllPoint()

    nodeNum = len(node)

    nodeDimension = node[0].size

    # Initialization
    phi = [1.0/clusterNum]*clusterNum
    mu = randomCenteroid(clusterNum, node)
    sigma = [rand(nodeDimension, nodeDimension)*min(x_max-x_min, y_max-y_min)
             * np.array([[1, 0], [0, 1]]) for x in range(clusterNum)]
    W = np.zeros((nodeNum, clusterNum))

    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(node[:, 0], node[:, 1], marker='^')
    ax[0].set_title("Original Data")
    draw_cluster(ax[0], x_min, x_max, y_min, y_max, clusterNum, phi, mu, sigma)

    for cnt in range(Maximum_iteration):
        phi0, mu0, sigma0, W0 = (phi, mu, sigma, W)

        # Expectation
        for i in range(nodeNum):
            W[i, :] = [
                phi[j]*multivariate_gaussian(node[i], mu[j], sigma[j]) for j in range(clusterNum)]
            W[i, :] /= np.sum(W[i, :])

        # Maximization
        phi = np.array([np.sum(W[:, j])/nodeNum for j in range(clusterNum)])

        mu = np.dot(W.T, node)
        mu = [mu[j]/np.sum(W[:, j]) for j in range(clusterNum)]

        for j in range(clusterNum):
            sigma[j] = np.zeros((nodeDimension, nodeDimension))
            for i in range(nodeNum):
                x = np.array([node[i]-mu[j]])
                sigma[j] += W[i][j]*x*x.T
            sigma[j] /= np.sum(W[:, j])

        if abs(reduce(lambda x, y: x**2+y**2, phi0)-reduce(lambda x, y: x**2+y**2, phi)) < deviation and np.dot((np.array(mu)-np.array(mu0)).T, np.array(mu)-np.array(mu0))[0, 0] < deviation:
        # if np.sum(np.abs(W - W0)) < deviation:
            break

    ax[1].scatter(node[:, 0], node[:, 1], marker='^')
    draw_cluster(ax[1], x_min, x_max, y_min, y_max, clusterNum, phi, mu, sigma)
    ax[1].set_title("after " + str(cnt+1) + " epoch(s)")

    print cnt, phi, mu, sigma
    fig.canvas.set_window_title(
        'GMM Clustering Results for ' + str(nodeNum) + ' nodes into ' + str(clusterNum) + ' clusters')
    plt.show()
    return


if __name__ == "__main__":
    Gaussian(3, 500, 5e-5)
