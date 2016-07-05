# -*- coding: utf-8 -*-
"""
Created on Tue Jul 05 16:05:34 2016

@Pycharm查看源码生命的小技巧：加入在编写这段程序的时候我们并不确定argsort（）是否为array对象的成员函数，
我们选中这个函数然后 右键 -> Go to -> Declaration，这样就会跳转到argsort（）函数的声明代码片中，
通过查看代码的从属关系能够确认array类中确实包含这个成员函数  


@author: lnn
"""
#单训练样本（每类只有一个训练样本）的分类问题，KNN的K值应该设定为1
import numpy as np  #科学计算包模块
import operator  #运算符模块

def createDataSet():
    group = np.array ([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]]) #通过array（）函数构造并初始化numpy的矩阵对象时，要保证只有一个参数
    labels = ['A','A','B','B']
    return group, labels

def classify0(inX,dataSet,labels,k):
    dataSetSize = np.shape(dataSet)[0] # shape[0]获取数据的行数，shape[1]求列数
    diffMat = np.tile(inX , (dataSetSize,1))-dataSet # tile（）函数是numpy的矩阵扩展函数
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort() #调用矩阵对象的排序成员函数argsort（）对距离进行升序排序
    

    classCount={}  #classCount字典 被分解为元组列表
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True) #逆序
    # 按key大小排列字典  指定函数针对字典中第二维元素进行排序
    return sortedClassCount[0][0] # 返回发生频率最高的元素标签
    
    
if __name__ == '__main__':
    data, labels = createDataSet()
    print classify0([1,1],data,labels,3) 
    
