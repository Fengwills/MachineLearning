import numpy as np
from sklearn.metrics import adjusted_rand_score
import random
iris_data_path = '../data/iris/iris.data'
heart_data_path = '../data/heart/heart.dat'
#load the dataset
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
# with open(heart_data_path,'r') as f:
#     heart_data = [line.strip().split(' ') for line in f.readlines()]
#     data_X = [x[:-1] for x in heart_data if x ]
#     data_X = np.array(data_X,dtype=float)
#     data_Y = [y[-1] for y in heart_data]
#     for index,i in enumerate(data_Y):
#         data_Y[index] = float(i)-1
#     data_Y = np.array(data_Y)
# Caculate Gaussian function
def Gaussian(data,mean,cov):
    dim = np.shape(cov)[0]   # 计算维度
    covdet = np.linalg.det(cov) # 计算|cov|
    covinv = np.linalg.inv(cov) # 计算cov的逆
    if covdet==0:              # 以防行列式为0
        covdet = np.linalg.det(cov+np.eye(dim)*0.01)
        covinv = np.linalg.inv(cov+np.eye(dim)*0.01)
    m = data - mean
    z = -0.5 * np.dot(np.dot(m, covinv),m)    # 计算exp()里的值
    return 1.0/(np.power(np.power(2*np.pi,dim)*abs(covdet),0.5))*np.exp(z)  # 返回概率密度值
# 用于判断初始聚类簇中的means是否距离离得比较近
def isdistance(means,criterion=0.03):
     K=len(means)
     for i in range(K):
         for j in range(i+1,K):
             if criterion>np.linalg.norm(means[i]-means[j]):
                 return False
     return True


# 获取最初的聚类中心
def GetInitialMeans(data,K,criterion):
    dim = data.shape[1]  # 数据的维度
    means = [[] for k in range(K)] # 存储均值
    minmax=[]
    for i in range(dim):
        minmax.append(np.array([min(data[:,i]),max(data[:,i])]))  # 存储每一维的最大最小值
    minmax=np.array(minmax)
    while True:
        for i in range(K):
            means[i]=[]
            for j in range(dim):
                 means[i].append(np.random.random()*(minmax[i][1]-minmax[i][0])+minmax[i][0] ) #随机产生means
            means[i]=np.array(means[i])

        if isdistance(means,criterion):
            break
    return means

# K均值算法，估计大约几个样本属于一个GMM
def Kmeans(data,K):
    N = data.shape[0]  # 样本数量
    dim = data.shape[1]  # 样本维度
    means = GetInitialMeans(data,K,15)
    means_old = [np.zeros(dim) for k in range(K)]
    # 收敛条件
    while np.sum([np.linalg.norm(means_old[k] - means[k]) for k in range(K)]) > 0.01:
        means_old = cp.deepcopy(means)
        numlog = [0] * K  # 存储属于某类的个数
        sumlog = [np.zeros(dim) for k in range(K)]
        # E步
        for i in range(N):
            dislog = [np.linalg.norm(data[i]-means[k]) for k in range(K)]
            tok = dislog.index(np.min(dislog))
            numlog[tok]+=1         # 属于该类的样本数量加1
            sumlog[tok]+=data[i]   # 存储属于该类的样本取值

        # M步
        for k in range(K):
            means[k]=1.0 / numlog[k] * sumlog[k]
    return means

def GMM(data,K):
    N = data.shape[0]
    dim = data.shape[1]
    means= Kmeans(data,K)
    convs=[0]*K
    # 初始方差等于整体data的方差
    for i in range(K):
        convs[i]=np.cov(data.T)
    pis = [1.0/K] * K
    gammas = [np.zeros(K) for i in range(N)]
    loglikelyhood = 0
    oldloglikelyhood = 1

    while np.abs(loglikelyhood - oldloglikelyhood) > 0.0001:
        oldloglikelyhood = loglikelyhood

        # E步
        for i in range(N):
            res = [pis[k] * Gaussian(data[i],means[k],convs[k]) for k in range(K)]
            sumres = np.sum(res)
            for k in range(K):           # gamma表示第n个样本属于第k个混合高斯的概率
                gammas[i][k] = res[k] / sumres
        # M步
        for k in range(K):
            Nk = np.sum([gammas[n][k] for n in range(N)])  # N[k] 表示N个样本中有多少属于第k个高斯
            pis[k] = 1.0 * Nk/N
            means[k] = (1.0/Nk)*np.sum([gammas[n][k] * data[n] for n in range(N)],axis=0)
            xdiffs = data - means[k]
            convs[k] = (1.0/ Nk)*np.sum([gammas[n][k]* xdiffs[n].reshape(dim,1) * xdiffs[n] for  n in range(N)],axis=0)
        # 计算最大似然函数
        loglikelyhood = np.sum(
            [np.log(np.sum([pis[k] * Gaussian(data[n], means[k], convs[k]) for k in range(K)])) for n in range(N)])
        print (means)
        print (loglikelyhood)
GMM(data_X,3)
