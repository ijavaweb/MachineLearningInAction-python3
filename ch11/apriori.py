from numpy import *


def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))

def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in list(Ck):
            if can.issubset(tid):
                if not can in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt.keys():
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData
def aprioriGen(LK,k):
    retList=[]
    lenLk=len(LK)
    for i in range(lenLk):
        for j in range(i+1,lenLk):
            L1=list(LK[i])[:k-2]
            L2=list(LK[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1==L2:
                retList.append(LK[i] | LK[j])
    return retList

def apriori(dataSet,minSupport=0.5):
    C1=createC1(dataSet)
    D=list(map(set,dataSet))
    L1,supportData=scanD(D,C1,minSupport)
    L=[L1]
    k=2
    while len(L[k-2])>0:
        Ck=aprioriGen(L[k-2],k)
        LK,supK=scanD(D,Ck,minSupport)
        supportData.update(supK)
        L.append(LK)
        k+=1
    return L,supportData


# 关联规则生成
def calcConf(freqSet,H,supportData,br1,minConf=0.7):
    prunedH=[]
    for conseq in H:
        conf=supportData[freqSet]/supportData[freqSet-conseq]
        if conf > minConf:
            br1.append((freqSet-conseq,conseq,conf))
            prunedH.append(conseq)
    return prunedH
def rulesFromConseq(freqSet,H,supportData,br1,minConf=0.7):
    m=len(H[0])
    if len(freqSet)>(m+1):
        Hmp1=aprioriGen(H,m+1)
        Hmp1=calcConf(freqSet,Hmp1,supportData,br1,minConf)
        if len(Hmp1)>1:
            rulesFromConseq(freqSet,Hmp1,supportData,br1,minConf)
def generateRules(L,supportData,minConf=0.7):
    bigRuleList=[]
    for i in range(1,len(L)):
        for freqSet in L[i]:
            H1=[frozenset([item]) for item in freqSet]
            if i>1:
                rulesFromConseq(freqSet,H1,supportData,bigRuleList,minConf)
            else:
                calcConf(freqSet,H1,supportData,bigRuleList,minConf)
    return bigRuleList