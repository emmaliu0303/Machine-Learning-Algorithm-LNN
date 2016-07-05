# -*- coding: utf-8 -*-
"""
Created on Tue Jul 05 19:04:49 2016
@使用Matplotlib创建散点图
  k-邻近改进约会网站的配对效果


@author: lnn
"""
import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt
import sys 

"""
 从文本文件解析数据:
"""
def loadFileMatrix(filename,delimiter = "\t"):

    fr = open (filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)  # 1,首先知道文本包含多少行
    returnMat = np.zeros((numberOfLines,3)) #创建以0填充的要返回的Numpy矩阵 行数*列数的维度，3可变化
    classLabelVector = []
    index = 0
    for line in arrayOLines:   #循环处理每行数据
        line = line.strip()   #截取函数，截掉回车符
        listFromLine = line.split(delimiter)  #用tab字符将整行数据分割成一个元素的列表
        returnMat[index,:] = listFromLine[0:3]   #取前三个元素存储到特征矩阵中
        classLabelVector.append(int(listFromLine[-1]))   # 索引值为-1时，表示列表中的最后一列元素（负索引）
                                                         # append() 方法用于在列表末尾添加新的对象。
        index +=1
    return returnMat,classLabelVector        

"""
 对数字特征值进行归一化（标准化）
"""
def autoNorm(dataSet):
    minVals = dataSet.min(0) # 参数0使得函数可以从每列中选取最小的值，而不是当前行的最小值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet)) #创建新的返回矩阵
    m = np.shape(dataSet)[0]  #shape[0]求行数，shape[1]求列数
    normDataSet = dataSet - np.tile(minVals , (m,1)) #tile函数将矩阵复制成与输入矩阵同样规模的矩阵 因为特征值矩阵是
    normDataSet = normDataSet/np.tile(ranges,(m,1)) #1000*3 而 minVals 和 ranges都是1*3维矩阵 重复矩阵 在行上m次，在列上1次
    return normDataSet,ranges,minVals               


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

def varDatingClassTest(normDataSet,datingLabels):
     hoRatio = 0.10 # 设定验证数据为10%
     m = np.shape(normDataSet)[0]  
     numTestVecs = int (m*hoRatio)
     errorCount = 0.0
     for i in range(numTestVecs):
       classifierResult = classify0(normDataSet[i,:],normDataSet[numTestVecs:m,:],datingLabels[numTestVecs:m],3) 
       if(classifierResult != datingLabels[i]):
           errorCount +=1.0
     return errorCount / float(numTestVecs)
        


"""    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:,1],datingDataMat[:,2], 15.0* np.array(datingLabels), 15.0* np.array(datingLabels))
    plt.show()  # 散点图使用矩阵datingDataMat的第二，第三列所有行的数据 均从0开始   
    print normDataSet
    #print datingDataMat,datingLabels
"""
if __name__ == '__main__':
    datingDataMat,datingLabels = loadFileMatrix ('.\\datingTestSet2.txt')
    normDataSet,ranges,minVals = autoNorm(datingDataMat)
    talError = varDatingClassTest(normDataSet,datingLabels)
    print "the total error rate is : %f" %talError
        













    