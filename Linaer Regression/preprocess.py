# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
# # 读取数据
# data = pd.read_csv('data/kaggle_hourse_price_train.csv')
# print(data.shape)
# # 丢弃有缺失值的特征（列）
# data.dropna(axis = 1, inplace = True)
#
# # 只保留整数的特征
# data = data[[col for col in data.dtypes.index if data.dtypes[col] == 'int64']]
#
# # 保存处理后的结果到csv文件
# data.to_csv("kaggle房价预测处理后数据.csv", index = False)
# print(data.shape)
data = pd.read_csv('kaggle房价预测处理后数据.csv')
# corrmat = data.corr()#相关系数
# k = 10 #number of variables for heatmap
# corr = corrmat.nlargest(10,'SalePrice')
# print(corrmat.nlargest(10,'SalePrice'))
# cols = corrmat.nlargest(10,'SalePrice')['SalePrice'].index
# cm = np.corrcoef(data[cols].values.T)
# sns.set(font_scale=1.25)
#
# hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
# plt.show()
features = ['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF']
# quantitative = [f for f in data.columns if data.dtypes[f] != 'object']
f = pd.melt(data,value_vars=features)
g = sns.FacetGrid(f,col='variable',col_wrap=2,sharex=False,sharey=False)
g = g.map(sns.distplot,'value')
plt.show()