import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import random
import os


class MySVM(BaseEstimator):
    def __init__(self,C,epoches,kernel):
        self.numSamples = None
        self.alphas = None
        self.b = 0
        self.errorCache = None
        self.weight = None
        self.C = C
        self.epoches = epoches
        self.kernel = kernel
        self.datax=None
        self.datay=None
        pass
    def fit(self, datax, datay):
        # training
        iter = 0
        alphaPairChanged = 0
        self.datax = datax
        self.datay = datay
        self.numSamples = datax.shape[0]
        self.errorCache = np.zeros([self.numSamples,2])
        self.alphas = np.zeros([self.numSamples,1])
        K = np.zeros([self.numSamples,self.numSamples])
        for i in range(self.numSamples):
            k = self.kernelTrans(self.datax,self.datax[i],kernelOp=self.kernel)
            k = k.reshape(k.shape[0])
            K[:,i] = k
        while iter<self.epoches:
            alphaPairChanged=0
            for i in range(self.numSamples):
                alphaPairChanged+=self.SMO(i,K)
            print('--iter:%d entire set,alpha pairs changed:%d'%(iter,alphaPairChanged))

            iter+=1
        self.weight = np.dot(np.transpose(self.alphas*datay),datax)
    def SMO(self,i,kernel):

        a_i,x_i,y_i = float(self.alphas[i]),self.datax[i],float(self.datay[i])
        Ei = self.calcError(i)
        j,Ej = self.select_j(i, self.numSamples,Ei)
        a_j,x_j,y_j = float(self.alphas[j]),self.datax[j],float(self.datay[j])
        Ej = self.calcError(j)
        # k_ii,k_jj,k_ij = np.dot(x_i,np.transpose(x_i)),np.dot(x_j,np.transpose(x_j)),np.dot(x_i,np.transpose(x_j))
        k_ii,k_jj,k_ij = kernel[i,i],kernel[j,j],kernel[i,j]
        eta = k_ii+k_jj-2*k_ij
        if eta<=0:
            return 0
        a_j_old = a_j
        a_j_new = a_j_old+y_j*(Ei-Ej)/eta
        if y_i !=y_j:
            L = max(0,a_j-a_i)
            H = min(self.C,self.C+a_j-a_i)
        else:
            L = max(0,a_i+a_j-self.C)
            H = min(self.C,a_i+a_j)
        a_j_new = self.clip(a_j_new,L,H)
        if a_j_new-a_j_old<0.00001:
            self.updateError(j)
            return 0
        a_i_new = a_i+y_i*y_j*(a_j_old-a_j_new)
        self.updateError(i)
        self.alphas[i],self.alphas[j] = a_i_new,a_j_new
        #caculate bias
        b_i_new = -Ei+y_i*k_ii*(a_i_new-a_i)-y_j*k_ij*(a_j_new-a_j_old)+self.b
        b_j_new = -Ej+y_i*k_ij*(a_i_new-a_i)-y_j*k_jj*(a_j_new-a_j_old)+self.b
        if (0< a_i_new < self.C ):
            self.b = b_i_new
        elif 0<a_j_new<self.C:
            self.b = b_j_new
        else:
            self.b = (b_i_new+b_j_new)/2
        self.updateError(i)
        self.updateError(j)
        return 1
        pass

    def predict(self, datax):
        p = np.dot(datax,np.transpose(self.weight)) + self.b
        for index, i in enumerate(p):
            if i >0:
                p[index] = '1'
            else:
                p[index] = '-1'
        return p

    def kernelTrans(self, X, sampleX, kernelOp):
        # cpmpute K（train_x,x_i）
        # param X: sample metrics
        # param sampleX: one sample
        m = X.shape[0]  # sample nums
        K = np.zeros([m, 1])
        if kernelOp[0] == 'linear':  # linear kernel
            x = np.transpose(sampleX)
            K = np.dot(X, np.transpose(sampleX))
        elif kernelOp[0] == 'rbf':  # Gaussion kernel
            sigma = kernelOp[1]
            if sigma == 0: sigma = 1
            for i in range(m):
                deltaRow = X[i] - sampleX
                deltaRow = deltaRow.reshape([1,len(deltaRow)])
                K[i] = np.exp(np.dot(deltaRow, np.transpose(deltaRow)) / (-2.0 * sigma ** 2))
        return K

    def select_j(self, i, m,Ei):
        # random select j except i
        # l = list(range(m))
        # seq = l[: i] + l[i + 1:]
        # return random.choice(seq)
        maxK = 0
        maxStep = 0
        Ej=0
        validEcacheList = np.nonzero(self.errorCache[:,0])
        if len(validEcacheList)>1:# choose max Ei-Ek
            for k in validEcacheList:
                if k==i:
                    continue
                Ek = self.calcError(k)
                step = abs(Ei-Ek)
                if(step>maxStep):
                    maxK = k
                    maxStep = step
                    Ej=Ek
            return maxK,Ej
        else:#first loop random
            l = list(range(m))
            seq = l[:i]+l[i+1:]
            j = random.choice(seq)
            Ej = self.calcError(j)
            return j,Ej


    def clip(self, alpha, L, H):
        # control alpha in a range from L to H
        if alpha < L:
            return L
        elif alpha > H:
            return H
        else:
            return alpha

    def calcError(self, k):
        # calculate kth sample 's error
        # not use kernel function
        fxk = np.sum(np.dot(self.alphas * self.datay * self.datax, np.transpose(self.datax[k]))) + self.b
        # use kernel function
        # fxk = np.sum(self.alphas*datay*np.dot(np.transpose(datax,datax[k])))+self.b
        Ek = fxk - float(self.datay[k])
        return Ek
    def updateError(self,k):
        #update errorCache
        Ek = self.calcError(k)
        self.errorCache[k] = [1,Ek]

spam_data_path = '../data/spambase/spambase.data'
with open(spam_data_path, 'r') as f:
    spam_data = [line.strip().split(',') for line in f.readlines()]
    random.shuffle(spam_data)
    data_X = [x[:-1] for x in spam_data]
    data_X = np.array(data_X, dtype=float)
    data_Y = [y[-1] for y in spam_data]
    for index,i in enumerate(data_Y):
        if i == '0':
            data_Y[index] = '-1'
    data_Y = np.array(data_Y, dtype=float)
    data_Y = data_Y[:, np.newaxis]
    sc = StandardScaler()
    data_X = sc.fit_transform(data_X)
    print()
mySVM = MySVM(0.6,50,['rbf',1])
# mySVM = MySVM(0.6,50,['linear'])
predictions = cross_val_predict(mySVM,data_X,data_Y,cv=10)
accuracy = accuracy_score(data_Y,predictions)
precision = precision_score(data_Y,predictions)
recall = recall_score(data_Y,predictions)
f1 = f1_score(data_Y,predictions)
print("accuracy|precision|recall|f1")
print("%f|%f|%f|%f"%(accuracy,precision,recall,f1))
# C=0.6 epoches=50
# accuracy|precision|recall|f1
# 0.854379|0.873285|0.737452|0.799641
# C=0.3 epoches = 60
# accuracy|precision|recall|f1
# 0.888068|0.845213|0.876448|0.860547
# C=0.6 epoches=50, choose max Ei-Ej, linear kernel
# accuracy|precision|recall|f1
# 0.848511|0.747560|0.929399|0.828621