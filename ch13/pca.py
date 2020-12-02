import numpy as np
def loadDataSet(fileName,delim='\t'):
    fr=open(fileName)
    stringArr=[line.strip().split(delim) for line in fr.readlines()]
    datArr=[list(map(float,line))for line in stringArr]
    return np.mat(datArr)
def pca(dataMat,topNfeat=9999999):
    meanVals=np.mean(dataMat,axis=0)
    meanRemoved=dataMat-meanVals
    covMat=np.cov(meanRemoved,rowvar=0)
    eigVals,eigVects=np.linalg.eig(np.mat(covMat))
    eigValInd=np.argsort(eigVals)
    eigValInd=eigValInd[:-(topNfeat+1):-1]
    redEigVects=eigVects[:,eigValInd]
    lowDDataMat=meanRemoved*redEigVects
    reconMat=(lowDDataMat*redEigVects.T)+meanVals
    return lowDDataMat,reconMat

#替换数据集中NaN的数据
def replaceNanWithMean():
    dataMat=loadDataSet('secom.data',' ')
    numFeat=np.shape(dataMat)[1]
    for i in range(numFeat):
        meanVal=np.mean(dataMat[np.nonzero(~np.isnan(dataMat[:,i].A))[0],i])
        dataMat[np.nonzero(np.isnan(dataMat[:,i].A))[0],i]=meanVal
    return dataMat




#降维测试函数
def plotData():
    import matplotlib
    import matplotlib.pyplot as plt
    dataMat=loadDataSet('testSet.txt')
    lowDMat,reconMat=pca(dataMat,1)
    fig = plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(dataMat[:,0].flatten().A[0],dataMat[:,1].flatten().A[0],marker='^',s=90)
    ax.scatter(reconMat[:,0].flatten().A[0],reconMat[:,1].flatten().A[0],marker='o',s=50,c='red')
    fig.show()