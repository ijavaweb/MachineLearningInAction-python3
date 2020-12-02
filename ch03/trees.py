from numpy import *
import operator
import treePlotter
def createDataSet():
    dataSet=[[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels=['no surfacing','flippers']
    return dataSet,labels
def clacShannonEnt(dataSet):
    numEntries=len(dataSet)
    labelsCount={}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelsCount.keys():
            labelsCount[currentLabel]=0
        labelsCount[currentLabel]+=1
    shannonEnt=0.0
    for key in labelsCount:
        prob=float(labelsCount[key])/numEntries
        shannonEnt-=prob*log2(prob)
    return shannonEnt
def splitDataSet(dataSet,axis,value):
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
#计算数据集划分方式的熵，并选出最好的
def chooseBestFeatureToSplit(dataSet):
    numFeatures=len(dataSet[0])-1
    baseEntropy=clacShannonEnt(dataSet)
    bestInfoGain=0.0
    bestFeature=-1
    for i in range(numFeatures):
        featList=[example[i] for example in dataSet]
        uniqueVals=set(featList)
        newEntropy=0.0
        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet,i,value)
            prob=len(subDataSet)/float(len(dataSet))
            newEntropy+=prob*clacShannonEnt(subDataSet)
        infoGain=baseEntropy-newEntropy
        if infoGain>bestInfoGain:
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature
#选择出现最多的分类名称
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
        sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
        return sortedClassCount[0][0]
#构建决策树
def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    bestFeat=chooseBestFeatureToSplit(dataSet)
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}}
    del (labels[bestFeat])
    featValues=[example[bestFeat] for example in dataSet]
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree
# 使用决策树分类
def classify(inputTree,featLabels,testVec):
    firstStr=list(inputTree.keys())[0]
    secondDict=inputTree[firstStr]
    featIndex=featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex]==key:
            if type(secondDict[key]).__name__=='dict':
                classLabel=classify(secondDict[key],featLabels,testVec)
            else:
                classLabel=secondDict[key]
    return classLabel
#持久化决策树模型
def storeTree(inputTree,filename):
    import pickle
    fw=open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()
#从磁盘读出持久化的数据
def grabTree(filename):
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)
#隐形眼镜测试
def treeTest():
    fr=open('lenses.txt')
    lenses=[inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels=['age','prescript','astigmatic','tearRate']
    lensesTree=createTree(lenses,lensesLabels)
    treePlotter.createPlot(lensesTree)