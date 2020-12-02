from numpy import *
def loadDataSet(fileName):
    numFeat=len(open(fileName).readline().split('\t'))-1
    dataMat=[]
    labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

#标准回归
def standRegress(xArr,yArr):
    xMat=mat(xArr)
    yMat=mat(yArr).T
    xTx=xMat.T*xMat
    if linalg.det(xTx)==0.0:
        print("this matrix is singular,connot do inverse")
        return
    ws=xTx.I*(xMat.T*yMat)
    return ws

#局部加权线性回归
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat=mat(xArr)
    yMat=mat(yArr).T
    m=shape(xMat)[0]
    weights=mat(eye((m)))
    for j in range(m):
        diffMat=testPoint-xMat[j,:]
        weights[j,j]=exp((diffMat*diffMat.T)/(-2.0*k**2))
    xTx=xMat.T*(weights*xMat)
    if linalg.det(xTx)==0.0:
        return
    ws=xTx.I*(xMat.T*(weights*yMat))
    return testPoint*ws
def lwlrTest(testArr,xArr,yArr,k=1.0):
    m=shape(testArr)[0]
    yHat=zeros(m)
    for i in range(m):
        yHat[i]=lwlr(testArr[i],xArr,yArr,k)
    return yHat
#预测鲍鱼年龄
def rssError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum()
def testAbalone():
    abX,abY=loadDataSet('abalone.txt')
    yHat01=lwlrTest(abX[0:99],abX[0:99],abY[0:99],0.1)
    yHat1=lwlrTest(abX[0:99],abX[0:99],abY[0:99],1)
    yHat10=lwlrTest(abX[0:99],abX[0:99],abY[0:99],10)
    print(rssError(abY[0:99],yHat01.T))
    print(rssError(abY[0:99],yHat1.T))
    print(rssError(abY[0:99],yHat10.T))


#岭回归，用于特征数多于样本数，或者用于在估计中加入偏差
def ridgeRegress(xMat,yMat,lam=0.2):
    xTx=xMat.T*xMat
    denom=xTx+eye(shape(xMat)[1])*lam
    if linalg.det(denom)==0.0:
        return
    ws=denom.I*(xMat.T*yMat)
    return ws
def ridgeTest(xArr,yArr):
    xMat=mat(xArr)
    yMat=mat(yArr).T
    yMean=mean(yMat,0)
    yMat=yMat-yMean
    xMeans=mean(xMat,0)
    xVar=var(xMat,0)
    xMat=(xMat-xMeans)/xVar
    numTestPts=30
    wMat=zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws=ridgeRegress(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat
# 前向逐步线性回归
def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat=mat(xArr)
    yMat=mat(yArr).T
    yMean=mean(yMat,0)
    yMat=yMat-yMean
    xMat=regularize(xMat)
    m,n=shape(xMat)
    returnMat=zeros((numIt,n))
    ws=zeros((n,1))
    wsTest=ws.copy()
    wsMax=ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowesError=inf
        for j in range(n):
            for sign in [-1,1]:
                wsTest=ws.copy()
                wsTest[j]+=eps*sign
                yTest=xMat*wsTest
                rssE=rssError(yMat.A,yTest.A)
                if rssE<lowesError:
                    lowesError=rssE
                    wsMax=wsTest
        ws=wsTest.copy()
        returnMat[i,:]=ws.T
    return returnMat
def regularize(xMat):
    inMat=xMat.copy()
    inMean=mean(inMat,0)
    inVar=var(inMat,0)
    inMat=(inMat-inMean)/inVar
    return inMat



# 乐高价格预测
from time import sleep
import json
import urllib
def searchForSet(retX,retY,setNum,yr,numPce,origPrc):
    sleep(10)
    myAPIstr='get from code.google.com'
    searchURL='https"//www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' %(myAPIstr,setNum)
    pg=urllib.urlopen(searchURL)
    retDict=json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem=retDict['items'][i]
            if currItem['product']['condition']=='new':
                newFlag=1
            else:
                newFlag=0
            listOfInv=currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice=item['price']
                if sellingPrice>origPrc*0.5:
                    retX.append([yr,numPce,newFlag,origPrc])
                    retY.append(sellingPrice)
        except:
            print("problem with item:%d" %i)



