# -*- coding:utf-8 -*-
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import minmax_scale
class LinearRegression(BaseEstimator):
    def __init__(self):
        self.theta = None

    def fit(self,trainX,trainY):
        # trainX = trainX.values
        ones = np.ones(trainX.shape[0])
        trainX = np.insert(trainX,0,values=ones,axis=1)
        self.theta = np.random.randn(trainX.shape[1],1)
        alpha=0.01
        iters = 10000
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
        x = range(10000)
        plt.plot(x, total_cost)
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
data_X = minmax_scale(datax)
# data_Y = minmax_scale(datay)
predictions = cross_val_predict(Linear,data_X,datay,cv=10)
m = len(predictions)
mse = 1.0/m*np.sum(abs(predictions-datay))
rmse = math.sqrt(1.0/m*np.sum((predictions-datay)**2))
print(mse,rmse)
predictions.sort(axis=0)

