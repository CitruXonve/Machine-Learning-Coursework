# coding: utf-8

import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

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
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N

def combine_multivariate_gaussian(pos, funNum, mu, sigma, phi):
    res = np.zeros(multivariate_gaussian(pos,mu[0],sigma[0]).shape)
    for j in range(funNum):
        res = res + multivariate_gaussian(pos,mu[j],sigma[j])*phi[j]
    return res

global x_min, x_max, y_min, y_max
X, x_min, x_max, y_min, y_max = InitAllPoint()
print(max(X[:,0]))
print(min(X[:,1]))

clusterNum=3

global fig, ax
fig, ax = plt.subplots(1, 2)
ax[0].scatter(X[:,0], X[:,1], marker='^')
ax[0].set_title("Original Data")
ax[1].scatter(X[:,0], X[:,1], marker='^')


# np.identity(2)*2

def GMM(X,iterations):
    mean=np.random.randint(-2,5,size=(3,2))
    cov=[np.identity(2)*5]*3
    fraction = np.ones(3)/3
    
    x = np.arange(x_min - 1, x_max + 1, 0.1)
    y = np.arange(y_min -1, y_max+1, 0.1)
    x, y = np.meshgrid(x, y)
    # z = combine_multivariate_gaussian(np.array([x.ravel(), y.ravel()]).T,clusterNum,mean,cov,fraction)
    for j in range(clusterNum):
        z = multivariate_gaussian(np.array([x.ravel(), y.ravel()]).T,mean[j],cov[j])
        z = np.array([z]) 
        z = z.reshape(len(np.arange(y_min -1, y_max+1, 0.1)),-1)
        ax[0].contour(x, y, z, colors='orange', alpha = fraction[j])

    for _ in range(iterations):  
        ric=Expectation(X,mean,cov,fraction)
        mean, cov, fraction=Maximization(X,ric)

    # z = combine_multivariate_gaussian(np.array([x.ravel(), y.ravel()]).T,clusterNum,mean,cov,fraction)
    for j in range(clusterNum):
        z = multivariate_gaussian(np.array([x.ravel(), y.ravel()]).T,mean[j],cov[j])
        z = np.array([z]) 
        z = z.reshape(len(np.arange(y_min -1, y_max+1, 0.1)),-1)
        ax[1].contour(x, y, z, colors='orange', alpha = fraction[j])

    return mean,cov, fraction
    
def Expectation(X,mean,cov,fraction):
    r = np.zeros((len(X),3))
    for m,c,f,i in zip(mean,cov,fraction,range(len(r[0]))):
        distribution = multivariate_normal(mean=m,cov=c)
        total=0
        for a,b,c in zip(mean,cov,fraction):
            total=total+c* multivariate_normal(mean=a,cov=b).pdf(X)
        r[:,i] = f*distribution.pdf(X)/total
    return(r)

def Maximization(X,r):
    mean=[]
    cov=[]
    fraction=[]
    for c in range(len(r[0])):
        m_c = np.sum(r[:,c],axis=0)
        mean_c = (1/m_c)*np.sum(X*r[:,c].reshape(len(X),1),axis=0)
        mean.append(mean_c)
        cov.append(((1/m_c)*np.dot((np.array(r[:,c]).reshape(len(X),1)*(X-mean_c)).T,(X-mean_c))))
        fraction.append(m_c/np.sum(r)) 
    return mean,cov,fraction
    
mean,cov,fraction=GMM(X,1000)
print("Mean:  ",mean)
print("Covariance Matrix:   ",cov)
print(fraction, np.sum(fraction))
plt.show()