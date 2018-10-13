# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 16:52:22 2017

@author: Lenovo-Y430p
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
def loadDataSet():
    dataMat = []; labelMat = []
    try:
        fr = open('C:/Users/msi/Desktop/lr-master/ex2data1.txt')
    except IOError:
        print("请检查您的路径")
    else:
        for line in fr.readlines():
            lineArr = line.strip().split(',')
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
            labelMat.append(int(lineArr[2]))
        fr.close()
        return np.mat(dataMat),np.mat(labelMat)

def function(x):
    return x**2
def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

def function_deriv(g):
    #convert to NumPy matrix
    dataMatIn,classLabels=loadDataSet()
    dataMatrix = np.mat(dataMatIn) #convert to NumPy matrix
    labelMat = np.mat(classLabels).transpose()
    
    h = sigmoid(dataMatrix*g) #matrix mult
    error = (labelMat - h)#vector subtraction
    l=dataMatrix.T*error#matrix mult
    return l

#Identity Matrix
I = np.array( [ [1,0,0],
                [0,1,0],[0,0,1] ])

#Hessian Matrix set to Identity Matrix
B = I

#Hessian Matrix Inverse set to B
B_inv = B

#Pick initial point x_0
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    labelMat=labelMat.tolist()[0]
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0] 
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(20, 120, 10)
    y = (-weights[0]-weights[1]*x)/weights[2]
    y=y.tolist()
    ax.plot(x, y[0])
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()


def BFGS(x,y, iter):#BFGS拟牛顿法
    n = np.shape(x)[1]
    theta=np.ones((n,1))
    y=np.mat(y).T
    Bk=np.eye(n,n)
    grad_last = np.dot(x.T,sigmoid(np.dot(x,theta))-y)
    cost=[]
    for it in range(iter):
        pk = -1 * np.linalg.solve(Bk, grad_last)   #搜索方向
        rate=alphA(x,y,theta,pk)
        theta = theta + rate * pk
        grad= np.dot(x.T,sigmoid(np.dot(x,theta))-y)
        delta_k = rate * pk
        y_k = (grad - grad_last)
        Pk = y_k.dot(y_k.T) / (y_k.T.dot(delta_k))
        Qk= Bk.dot(delta_k).dot(delta_k.T).dot(Bk) / (delta_k.T.dot(Bk).dot(delta_k)) * (-1)
        Bk += Pk + Qk
        grad_last = grad
        print(np.sum(grad_last))
        cost.append(np.sum(grad_last))
    return theta,cost

def alphA(x,y,theta,pk): #选取前2000次迭代cost最小的alpha
    c=float("inf")
    t=theta
    for k in range(1,2000):
            a=1.0/k**2
            theta = t + a * pk
            f= np.sum(np.dot(x.T,sigmoid(np.dot(x,theta))-y))
            if abs(f)>c:
                break
            c=abs(f)
            alpha=a
    return alpha
x,y=loadDataSet()
x1,y1=BFGS(x,y, iter=1000)
plotBestFit(x1)
