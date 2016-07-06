# -*- coding: utf-8 -*-
"""
Created on Wed Jul 06 10:59:38 2016
 梯度上升法优化logistics regression
 



@author: lnn
"""
"""
 打开文件进行逐行读取
"""

import numpy as np
import operator
import matplotlib.pyplot as plt





def loadDataSet(filename):
    dataMat = []; labelMat = []  #定义列表
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split();
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])]) # 每行前两个值X1，X2为输入值，且设置X0的值为1.0
        labelMat.append(int (lineArr[2])) # 第三个值对应的是所属的类别标签
    return dataMat,labelMat
    
def sigmoid(inX): #inX是用于分类的输入向量
     return 1.0/(1+np.exp(-inX))
"""
  梯度上升法优化 gradAscent()
"""
def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose() #进行矩阵的转置
    m,n = np.shape(dataMatrix)
    alpha = 0.001  #学习率
    maxCycles = 500 #迭代次数
    weights = np.ones((n,1)) #生成一个元素均为1的n*1维矩阵
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h) 
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

"""
  随机梯度上升优化法 stoGradAscent()
"""
def stoGradAscent(dataMatrix, classLabels,numIter = 150):
    m,n = np.shape(dataMatrix) #列表的行数和列数
    weights = np.ones(n) #生成n个数字为1的数
    for j in range(numIter):             
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01   #每次迭代进行调整，调整方式自定义
            randIndex = int (np.random.uniform(0,len(dataIndex)))  #随机选取更新  减少周期性波动
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

"""
 画出决策边界函数


"""

def plotBestFit(dataMat,labelMat,weights):
    dataArr = np.array(dataMat)
    n = np.shape(dataMat)[0]
    xcord1=[]; ycord1=[]
    xcord2=[]; ycord2=[]
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2,ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

if __name__ == '__main__':
    dataArr,labelMat = loadDataSet('testSet.txt')
    weights = stoGradAscent (np.array(dataArr),labelMat)
    plotBestFit(dataArr,labelMat,weights)
    print weights
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    