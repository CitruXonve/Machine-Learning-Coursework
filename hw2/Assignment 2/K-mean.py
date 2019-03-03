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


def Distance(a, b):
    return norm(a - b)

def randomCenteroid(ClusterNum, f):
    x_min = min(f[:, 0])
    x_max = max(f[:, 0])
    y_min = min(f[:, 1])
    y_max = max(f[:, 1])
    p = rand(ClusterNum, 2)
    p = p * np.array([np.ones(ClusterNum) * (x_max - x_min), np.ones(ClusterNum) * (y_max - y_min)]
                     ).transpose() + np.array([np.ones(ClusterNum) * x_min, np.ones(ClusterNum) * y_min]).transpose()
    return p


def classify(nodes, Centeroid):
    N = len(nodes)
    M = len(Centeroid)
    idx = [0]*N
    ClusterSize=[0]*M
    for i in range(N):
        dist0 = norm(nodes[i] - Centeroid[0])
        for j in range(1, M):
            dist = norm(nodes[i] - Centeroid[j])
            if dist < dist0:
                dist0 = dist
                idx[i] = j
        ClusterSize[idx[i]]+=1
    return idx,ClusterSize


def generate_new_center(nodes, idx, ClusterSize):
    centers = np.zeros((np.size(ClusterSize),2))
    for i in range(np.sum(ClusterSize)):
        centers[idx[i]]+=nodes[i]

    for i in range(np.size(ClusterSize)):
        if ClusterSize[i]>0:
            centers[i]/=ClusterSize[i];
    return centers


def judge_center_difference(centers, new, deviation):
    return norm(new - centers) > deviation  

def draw_clusters(ax, nodes, idx, centers, colors, show_circles=True):
    ax.clear()
    for i in range(len(centers)):
        r = 0
        ax.scatter(centers[i][0], centers[i][1],
                   edgecolors=[0.8,0.1,0.1],  facecolors=colors[i])
        for index in [q for q in range(np.size(idx, 0)) if idx[q] == i]:
            ax.scatter(nodes[index][0], nodes[index]
                       [1], marker='^', facecolors=colors[i])
            r = max(r, norm(nodes[index] - centers[i]))
        if show_circles:
            draw_circle(ax, centers[i][0], centers[i]
                        [1], r, [colors[i]])


def draw_circle(ax, x0, y0, r, color):
    x = y = np.arange(-r - 1, r + 1, 0.1)
    x, y = np.meshgrid(x, y)
    ax.contour(x + x0, y + y0, x**2 + y**2,
               [r * r], colors=color, linestyles='dotted')


def InitAllPoint():
    res=[]
    f=open('clusters.txt','r');
    for line in f.readlines():
        res.append([float(x) for x in line.split(',')])
    return np.array(res)

def GetKMeanResult(Centeroid, f, deviation, Maximum_iteration):
    Last_idx = FindNearest(Centeroid, f)
    Cur_idx = Last_idx
    same_time = 0
    for iter in range(1, Maximum_iteration):
        Centeroid = UpdateCenter(f, Last_idx, np.size(Centeroid, 0))
        Last_idx = Cur_idx
        Cur_idx = FindNearest(Centeroid, f)
        if (np.sum(np.abs(Last_idx - Cur_idx)) < deviation):
            same_time += 1
        else:
            same_time = 0

        if same_time >= 3:
            break

        # break
    return Cur_idx, Centeroid, iter

def UpdateCenter(f, idx, ClusterNum):
    res = np.zeros((ClusterNum , 2))
    for i in range(0, ClusterNum):
        tmp=np.array([q for q in range(np.size(idx,0)) if idx[q] == i+1])
        if np.size(tmp)>0:
            clu = f[tmp,:]
            res[i] = np.sum(clu, 0) / np.size(clu, 0)
    return res


def FindNearest(CenterSet, f):
    idx = np.zeros(np.size(f, 0))
    for i in range(0, np.size(f, 0)):
        dis_set = np.zeros(np.size(CenterSet, 0))
        for j in range(0, np.size(CenterSet, 0)):
            dis_set[j] = Distance(CenterSet[j,:], f[i,:])

        min_idx = list(dis_set).index(min(dis_set))
        idx[i] = min_idx
    return idx

def runKmean(Centeroid,nodes, deviation, Maximum_iteration, ax, colors):
    idx, ClusterSize = classify(nodes, Centeroid)

    cnt = 0

    draw_clusters(ax, nodes, idx, Centeroid, colors, False)
    ax.set_title("after " + str(cnt) + " epoch(s), SSE=" + str(computeSSE(nodes, idx, Centeroid)))
    plt.savefig("after " + str(cnt) + " epoch(s).png")

    NewPoint = generate_new_center(nodes, idx, ClusterSize)

    while judge_center_difference(Centeroid, NewPoint, deviation) and cnt < Maximum_iteration:
        Centeroid=NewPoint
        idx, ClusterSize = classify(nodes, Centeroid)
        NewPoint = generate_new_center(nodes, idx, ClusterSize)
        cnt += 1

        draw_clusters(ax, nodes, idx, Centeroid, colors, False)
        ax.set_title("after " + str(cnt) + " epoch(s), SSE=" + str(computeSSE(nodes, idx, Centeroid)))
        plt.savefig("after " + str(cnt) + " epoch(s).png")

    return idx, Centeroid, cnt

def computeSSE(nodes,idx,Centeroid):
    res=0.0
    for i in range(np.size(nodes,0)):
        temp=Distance(Centeroid[idx[i]],nodes[i])
        res+=temp**2
    return res

def K_mean(ClusterNum, Maximum_iteration, deviation):

    nodes = InitAllPoint()

    initCenteroid = randomCenteroid(ClusterNum, nodes)

    idx=np.zeros(len(nodes))
    colors=np.ones((ClusterNum,3))*0.8

    fig, ax = plt.subplots(1, 2)
    draw_clusters(ax[0], nodes, idx, initCenteroid, colors, False)
    ax[0].set_title("Original Data")

    colors = rand(ClusterNum, 3) * 0.8

    classified, Centeroid, count=runKmean(initCenteroid, nodes, deviation, Maximum_iteration, ax[1], colors)
    SSE=computeSSE(nodes, classified, Centeroid)

    draw_clusters(ax[1], nodes, classified, Centeroid, colors, False)

    ax[1].set_title("after " + str(count) + " epoch(s), SSE=" + str(SSE))

    fig.canvas.set_window_title(
        'K-mean Convergence Results for ' + str(len(nodes)) + ' nodes into ' + str(ClusterNum) + ' clusters')
    plt.show()

    print Centeroid, count, SSE

    return count,SSE

if __name__ == "__main__":
    K_mean( 3, 100, 1e-5)
