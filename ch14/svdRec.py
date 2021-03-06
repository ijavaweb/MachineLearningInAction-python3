import numpy as np
import numpy.linalg as la
def loadExData():
    return [[0, 0, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 1, 0, 0]]
def loadExData2():
    return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
            [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
            [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
            [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
            [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
            [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
            [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
            [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

# 相似度计算

#欧拉相似度
def eulidSim(inA,inB):
        return 1.0/(1.0+la.norm(inA-inB))

# 皮尔逊相似度
def pearsSim(inA,inB):
        if len(inA)<3:
                return 1.0
        return 0.5+0.5*np.corrcoef(inA,inB,rowvar=0)[0][1]
#余弦相似度
def cosSim(inA,inB):
        num=float(inA.T*inB)
        denom=la.norm(inA)*la.norm(inB)
        return 0.5+0.5*(num/denom)
# 基于物品相似度推荐引擎
def standEst(dataMat,user,simMeas,item):
        n=np.shape(dataMat)[1]
        simTotal=0.0
        ratSimTotal=0.0
        for j in range(n):
                userRating=dataMat[user,j]
                if userRating==0:
                        continue
                overLap=np.nonzero(np.logical_and(dataMat[:,item].A>0,dataMat[:,j].A>0))[0]
                if len(overLap)==0:
                        similarity=0
                else:
                        similarity=simMeas(dataMat[overLap,item],dataMat[overLap,j])
                simTotal+=similarity
                ratSimTotal+=similarity*userRating
        if simTotal==0:
                return 0
        else:
                return ratSimTotal/simTotal
def recommend(dataMat,user,N=3,simMeas=cosSim,estMethod=standEst):
        unratedItems=np.nonzero(dataMat[user,:].A==0)[1]
        if len(unratedItems)==0:
                return 'you rated everything'
        itemScores=[]
        for item in unratedItems:
                estimatedScore=estMethod(dataMat,user,simMeas,item)
                itemScores.append((item,estimatedScore))
        return sorted(itemScores,key=lambda p:p[1],reverse=True)[:N]

#基于SVD的评分估计
def svdEst(dataMat,user,sigMeas,item):
        n=np.shape(dataMat)[1]
        simTotal=0.0
        ratSimTotal=0.0
        U,Sigma,VT=la.svd(dataMat)
        Sig4=np.mat(np.eye(4)*Sigma[:4])
        xformedItems=dataMat.T*U[:,:4]*Sig4.I
        for j in range(n):
                userRating=dataMat[user,j]
                if userRating==0 or j==item:
                        continue
                similarity=sigMeas(xformedItems[item,:].T,xformedItems[j,:].T)
                simTotal+=similarity
                ratSimTotal+=similarity*userRating
        if simTotal==0:
                return 0
        else:
                return ratSimTotal/simTotal


# 基于SVD的图像压缩
def printMat(inMat,thresh=0.8):
        for i in range(32):
                for k in range(32):
                        if float(inMat[i,k])>thresh:
                                print(1,end="")
                        else:
                                print(0,end="")
                print('')
def imgCompress(numSV=3,thresh=0.8):
        my1=[]
        for line in open('0_5.txt').readlines():
                newRow=[]
                for i in range(32):
                        newRow.append(int(line[i]))
                my1.append(newRow)
        myMat=np.mat(my1)
        print("****original matrix******")
        printMat(myMat,thresh)
        U,Sigma,VT=la.svd(myMat)
        SigRecon=np.mat(np.zeros((numSV,numSV)))
        for k in range(numSV):
                SigRecon[k,k]=Sigma[k]
        reconMat=U[:,:numSV]*SigRecon*VT[:numSV,:]
        print("****reconstructed matrix using %d singular values******" %numSV)
        printMat(reconMat,thresh)