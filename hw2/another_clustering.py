# -*- coding: utf-8 -*-
# Python version 2.7
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
import math
from random import randint
from numpy.random import rand
from numpy.linalg import norm


# def Distance(a, b):
#     return norm(a - b)

def randomCenteroid(clusterNum, f):
    x_min = min(f[:, 0])
    x_max = max(f[:, 0])
    y_min = min(f[:, 1])
    y_max = max(f[:, 1])
    p = rand(clusterNum, 2)
    p = p * np.array([np.ones(clusterNum) * (x_max - x_min), np.ones(clusterNum) * (y_max - y_min)]
                     ).transpose() + np.array([np.ones(clusterNum) * x_min, np.ones(clusterNum) * y_min]).transpose()
    return p


def draw_circle(ax, x0, y0, r, color):
    x = y = np.arange(-r - 1, r + 1, 0.1)
    x, y = np.meshgrid(x, y)
    ax.contour(x + x0, y + y0, x**2 + y**2,
               [r * r], colors=color, linestyles='dotted')

def draw_ellipse(ax, x, y, z):
    x = y = np.arange(-r - 1, r + 1, 0.1)
    x, y = np.meshgrid(x, y)
    ax.contour(x, y, z,
               [0], colors=color, linestyles='dotted')


def InitAllPoint():
    res=[]
    f=open('clusters.txt','r');
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

# def combine_multivariate_gaussian(pos, funNum, mu, sigma, phi):
#     res = np.zeros(multivariate_gaussian(pos,mu[0],sigma[0]).shape)
#     for j in range(funNum):
#         res = res + multivariate_gaussian(pos,mu[j],sigma[j])*phi[j]
#     return res

def Gaussian(clusterNum, Maximum_iteration, deviation):

    # ClusterSize = [25,16,24,17,18];
    # m_idx = np.zeros(sum(ClusterSize))
    # tt = np.append([0], np.cumsum(ClusterSize))
    # for i in range(clusterNum):
    #     m_idx[tt[i]:tt[i + 1]] = np.ones(ClusterSize[i]) * i

    node, x_min, x_max, y_min, y_max = InitAllPoint()
    # node = rand(NodeNum, 2) * 200 - 100
    # print node

    initCenteroid = randomCenteroid(clusterNum, node)
    # initCenteroid = InitByKMeanPlusPlus(clusterNum, node)
    # print initCenteroid

    # idx=np.zeros(sum(ClusterSize))
    # idx=np.zeros(len(node))
    # colors=np.ones((clusterNum,3))*0.8

    fig, ax = plt.subplots(1, 2)
    # draw_clusters(ax[0], node, clusters, centers, colors, False)
    # draw_clusters(ax[0], node, idx, initCenteroid, colors, False)
    ax[0].scatter(node[:,0], node[:,1], marker='^')
    ax[0].set_title("Original Data")

    # colors = rand(clusterNum, 3) * 0.5

    nodeNum=len(node)

    nodeDimension=node[0].size

    phi=[1.0/clusterNum]*clusterNum

    mu=randomCenteroid(clusterNum, node)

    # sigma=[np.array([[1,0],[0,1]])]*clusterNum
    sigma=[rand(nodeDimension,nodeDimension)*min(x_max-x_min,y_max-y_min)*np.array([[1,0],[0,1]]) for x in range(clusterNum)]

    # print phi, mu, sigma

    x = np.arange(x_min - 1, x_max + 1, 0.1)
    y = np.arange(y_min -1, y_max+1, 0.1)
    x, y = np.meshgrid(x, y)
    # z = combine_multivariate_gaussian(np.array([x.ravel(), y.ravel()]).T,clusterNum,mu,sigma,phi)
    for j in range(clusterNum):
        z = multivariate_gaussian(np.array([x.ravel(), y.ravel()]).T,mu[j],sigma[j])
        z = np.array([z]) 
        z = z.reshape(len(np.arange(y_min -1, y_max+1, 0.1)),-1)
        ax[0].contour(x, y, z, colors='orange', alpha = phi[j])

    for cnt in range(Maximum_iteration):
        phi0, mu0, sigma0 = (phi, mu, sigma)

        #Expectation
        W=np.zeros((nodeNum,clusterNum))
        for i in range(nodeNum):
            # for j in range(clusterNum):
            #     W[i][j]=phi[j]*multivariate_gaussian(node[i],mu[j],sigma[j])
            W[i,:] = [phi[j]*multivariate_gaussian(node[i],mu[j],sigma[j]) for j in range(clusterNum)]
            W[i,:]/= np.sum(W[i,:])

        #Maximization
        phi=np.array([np.sum(W[:,j])/nodeNum for j in range(clusterNum)])

        mu=np.dot(W.T,node)
        mu=[mu[j]/np.sum(W[:,j]) for j in range(clusterNum)]
        # print 'mu:',np.array(mu).shape

        for j in range(clusterNum):
            sigma[j]=np.zeros((nodeDimension,nodeDimension))
            # print W[:,j].shape,node.shape,np.array([mu[j]]*nodeNum).shape
            # matrix = node - np.array([mu[j]]*nodeNum)
            # sigma[j]=np.dot(W[:,j],np.dot(matrix,matrix.T))
            for i in range(nodeNum):
                x = np.array([node[i]-mu[j]])
                sigma[j]+= W[i][j]*x*x.T
            sigma[j]/=np.sum(W[:,j])

        # print np.array(sigma).shape
        # print phi, mu, sigma

        x = np.arange(x_min - 1, x_max + 1, 0.1)
        y = np.arange(y_min -1, y_max+1, 0.1)
        x, y = np.meshgrid(x, y)
        # z = combine_multivariate_gaussian(np.array([x.ravel(), y.ravel()]).T,clusterNum,mu,sigma,phi)
        ax[1].clear()
        ax[1].scatter(node[:,0], node[:,1], marker='^')
        for j in range(clusterNum):
            z = multivariate_gaussian(np.array([x.ravel(), y.ravel()]).T,mu[j],sigma[j])
            z = np.array([z]) 
            z = z.reshape(len(np.arange(y_min -1, y_max+1, 0.1)),-1)
            ax[1].contour(x, y, z, colors='orange', alpha = phi[j])
        # plt.savefig(str(cnt)+" epoch(s).png")

        # print abs(reduce(lambda x,y:x**2+y**2,phi0)-reduce(lambda x,y:x**2+y**2,phi))

        if abs(reduce(lambda x,y:x**2+y**2,phi0)-reduce(lambda x,y:x**2+y**2,phi))<deviation and np.dot((np.array(mu)-np.array(mu0)).T,np.array(mu)-np.array(mu0))[0,0]<deviation:
            break

    print cnt, phi, mu, sigma
    plt.show()
    return

    ax[1].set_title("after " + str(count) + " epoch(s), SSE=" + str(SSE))

    fig.canvas.set_window_title(
        'K-mean Convergence Results for ' + str(len(node)) + ' nodes into ' + str(clusterNum) + ' clusters')
    

    print Centeroid, count, SSE

    return count,SSE

if __name__ == "__main__":
    Gaussian( 3, 500, 5e-5)
