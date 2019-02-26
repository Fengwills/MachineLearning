# -*- coding:utf-8 -*-

import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import cross_val_predict
class LinearRegression(BaseEstimator):
    def __init__(self):
        self.theta = None
    def Gaussian(self,x,means,var):
        if var==0:
            var = 1e-6
        exponent = math.exp(-((math.pow(x-means,2)/(2*var))))
        return (1/math.sqrt(2*math.pi*var))*exponent
    def fit(self,trainX,trainY):
        ones = np.ones(trainX.shape[0])
        trainX = np.insert(trainX,0,values=ones,axis=1)
        means1 = np.mean(trainX[:,1])
        var1 = np.var(trainX[:,1])
        means2 = np.mean(trainX[:,2])
        var2 = np.var(trainX[:,2])
        means3 = np.mean(trainX[:,3])
        var3 = np.var(trainX[:,3])
        means4 = np.mean(trainX[:,4])
        var4 = np.var(trainX[:,4])
        means5 = np.mean(trainX[:,5])
        var5 = np.var(trainX[:,5])
        for i in range(trainX.shape[0]):
            trainX[i, 1] = self.Gaussian(trainX[i, 1], means1, var1)
            trainX[i,2] = self.Gaussian(trainX[i,2],means2,var2)
            trainX[i, 3] = self.Gaussian(trainX[i, 3], means3, var3)
            trainX[i,4] = self.Gaussian(trainX[i,4], means4, var4)
            trainX[i,5] = self.Gaussian(trainX[i,5], means5, var5)
        self.theta = np.random.randn(trainX.shape[1],1)
        alpha=0.01
        iters = 100000
        self.theta = self.gradientDescent(trainX,trainY,alpha,iters)
        pass

    #alpha 学习率 iters迭代次数 theta权重矩阵
    def gradientDescent(self,X,y,alpha,iters):
        total_cost = []
        m = X.shape[0]
        for i in range(iters):
            hypothesis = np.dot(X,self.theta)
            loss = (hypothesis-y)
            gradient = np.dot(X.T,loss)/m
            self.theta = self.theta-alpha*gradient
            cost = 1.0/2*m*np.sum(np.square(np.dot(X,self.theta)-y))
            total_cost.append(cost)
            print("cost:%f"%cost)
        x = range(100000)

        plt.plot(x,total_cost)
        plt.xlabel("iters")
        plt.ylabel("loss")
        plt.show()

        return self.theta
        pass
    def predict(self,testX):
        ones = np.ones(testX.shape[0])
        testX = np.insert(testX, 0, values=ones, axis=1)
        return np.dot(testX,self.theta)

data = pd.read_csv('kaggle房价预测处理后数据.csv')
corrmat = data.corr()#相关系数
k = 5 #number of variables for heatmap
cols = corrmat.nlargest(k+1,'SalePrice')['SalePrice'].index
data_X = data[cols].drop(['SalePrice'],axis=1)
data_Y = data['SalePrice']
data_X = data_X.astype(float)
data_Y = data_Y.astype(float)
datax = data_X.as_matrix()
datay = data_Y.as_matrix().reshape(len(data_Y),1)

Linear = LinearRegression()
# trainx,testx,trainy,testy = train_test_split(datax,datay,test_size=0.3,random_state=1)
# datax = minmax_scale(trainx)
predictions = cross_val_predict(Linear,datax,datay,cv=10)
# Linear.fit(trainx,trainy)
# predictions = Linear.predict(testx)

m = len(predictions)

test = predictions-datay
mse = 1.0/m*np.sum(abs(predictions-datay))
rmse = math.sqrt(1.0/m*np.sum((predictions-datay)**2))
print(mse,rmse)
