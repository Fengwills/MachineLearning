import sys
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn import manifold
import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.metrics import fowlkes_mallows_score
iris_data_path = '../data/iris/iris.data'
heart_data_path = '../data/heart/heart.dat'
#'矩阵的逆矩阵需要的库'
from numpy.linalg import *
def plotResult(data,re_label,cluster_num):
    tsne = manifold.TSNE(n_components=2,init='pca',random_state=1)
    X_tsne = tsne.fit_transform(data)
    colors = ['b','g','r','#e24fff']
    max_i = max(X_tsne[:,0])
    min_i = min(X_tsne[:,0])
    max_j = max(X_tsne[:,1])
    min_j = min(X_tsne[:,1])
    for i in range(cluster_num):
        index = np.nonzero(re_label==i)[0]
        x0 = X_tsne[index,0]
        x1 = X_tsne[index,1]
        for j in range(len(x0)):
            plt.scatter(x0[j],x1[j],color = colors[i])
    plt.axis([min_i,max_i,min_j,max_j])
    plt.show()
def gmm(X,K,threshold=1e-15):
    # threshold  = 1e-15
    N,D = np.shape(X)
    randV = random.sample(range(1,N),K)
    centroids = X[randV]
    pMiu,pPi,pSigma = inti_params(centroids,K,X,N,D);
    Lprev = -np.inf
    while True:
 #       'Estiamtion Step'
        Px = calc_prop(X,N,K,pMiu,pSigma,threshold,D)
        pGamma = Px * np.tile(pPi,(N,1))
        pGamma = pGamma / np.tile((np.sum(pGamma,axis=1)),(K,1)).T
  #      'Maximization Step'
        Nk = np.sum(pGamma,axis=0)
        pMiu = np.dot(np.dot(np.diag(1 / Nk),pGamma.T),X)
        pPi = Nk / N
        for kk in range(K):
            Xshift = X - np.tile(pMiu[kk],(N,1))
            pSigma[:,:,kk] = (np.dot(np.dot(Xshift.T,np.diag(pGamma[:,kk])),Xshift)) / Nk[kk]

   #     'check for convergence'            
        L = np.sum(np.log(np.dot(Px,pPi.T)))
        if L-Lprev<threshold:
            break
        Lprev = L

    return Px

def inti_params(centroids,K,X,N,D):
    pMiu = centroids
    pPi = np.zeros((1,K))
    pSigma = np.zeros((D,D,K))
    distmat = np.tile(np.sum(X * X,axis=1),(K,1)).T \
    + np.tile(np.sum(pMiu * pMiu,axis = 1).T,(N,1)) \
    - 2 * np.dot(X,pMiu.T)
    labels = np.argmin(distmat,axis=1)

    for k in range(K):
        Xk = X[labels==k]
        pPi[0][k] = float(np.shape(Xk)[0]) / N # 样本数除以 N 得到概率
        pSigma[:,:,k] = np.cov(Xk.T)
    return pMiu,pPi,pSigma

    #'计算概率'
def calc_prop(X,N,K,pMiu,pSigma,threshold,D):
    Px = np.zeros((N,K))
    for k in range(K):
        Xshift = X - np.tile(pMiu[k],(N,1))
        # inv_pSigma = inv(pSigma[:,:,k]) \
        # + np.diag(np.tile(threshold,(1,np.ndim(pSigma[:,:,k]))))
        m = 1e-6
        t = pSigma[:, :, k].shape[1] * m
        test = np.eye(pSigma[:,:,k].shape[0])
        inv_pSigma = inv(pSigma[:,:,k]+m*test) \
        + np.diag(np.tile(threshold,(1,np.ndim(pSigma[:,:,k]))))
        tmp = np.sum(np.dot(Xshift,inv_pSigma) * Xshift,axis=1)
        coef = (2*np.pi)**(-D/2) * np.sqrt(np.linalg.det(inv_pSigma))
        Px[:,k] = coef * np.exp(-0.5 * tmp)
    return Px
def plotK(x,scores):
    plt.plot(x,scores)
    plt.xlabel("value of nClusters")
    plt.ylabel("adjusted_rand_score")
    plt.show()
def test():
    with open(iris_data_path,'r') as f:
        iris_data = [line.strip().split(',') for line in f.readlines()]
        data_X = [x[:-1] for x in iris_data if x]
        data_X = np.array(data_X,dtype=float)
        data_Y = [y[-1] for y in iris_data]
        for index,y in enumerate(data_Y):
            if y=='Iris-setosa':
                data_Y[index]=0
            elif y=='Iris-versicolor':
                data_Y[index] =1
            else:
                data_Y[index]=2
    X = data_X
    ppx = gmm(X,3)
    index = np.argmax(ppx,axis=1)
    ari = adjusted_rand_score(data_Y,index)
    homogeneity_score, completeness_score, v_measure_score = homogeneity_completeness_v_measure(data_Y,index)

    fmi_score = fowlkes_mallows_score(data_Y,index)
    print(ari,homogeneity_score,completeness_score,v_measure_score)
    print(fmi_score)
#   #iris-data 0.9038742317748124 0.8983263672602775 0.9010648908640206 0.8996935451597475 0.9355985958131775
    #heart-data  select real data 0.22167517444059512 0.16202649228984586 0.1643473829736618 0.16317868554116924
    # all data  -0.0035508279416603273 0.00019433966114699227 0.0003182578614206825 0.00024132042088711506
    # original select data  0.22167517444059512 0.16202649228984586 0.1643473829736618 0.16317868554116924
    # all data -0.004400028701202652 0.00047097880968955693 0.0007596579893208739 0.0005814596408448124
    threshold = [2,3,4,5,6,7]
    aris = []
    for th in threshold:
        ppx = gmm(X,th)
        index = np.argmax(ppx,axis=1)
        ari = adjusted_rand_score(data_Y,index)
        aris.append(ari)
    plotK(threshold,aris)
    
if __name__ == '__main__':
    test()