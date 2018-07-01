import numpy as np
import pandas as pd
import operator

###classify0 is the kNN classifier algorithm
def classify0(inX,dataSet,labels,k):
    '''
       inX is the vector you want to classify
       dataSet is the training dataset
       labels is the training labelset
       k is the parameter
    '''
    dataSetSize = dataSet.shape[0]
    ###Euclidean distance 
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    ########
    sortedDistIndicies = distances.argsort()
    classCount = {}
    ###count the labels for the shortest k samples
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    ########
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

#use pandas to read data
def file2dataframe(filename):
    data = pd.read_table(filename,header = None,encoding = 'gb2312',delim_whitespace = True)
    ###supposing raw data has 4 columns
    returnMat = data.loc[:,[0,1,2]].values
    classLabelVector = data.loc[:,3].values
    return returnMat,classLabelVector

#normalization
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals,(m,1))
    normDataSet = normDataSet/np.tile(ranges,(m,1))
    return normDataSet,ranges,minVals


