import numpy as np
import random
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.metrics import fowlkes_mallows_score
from sklearn import manifold
import matplotlib.pyplot as plt

iris_data_path='../data/iris/iris.data'
heart_data_path = '../data/heart/heart.dat'
# Load the dataset
with open(iris_data_path,'r') as f:
    iris_data = [line.strip().split(',') for line in f.readlines()]
    data_X = [x[:-1] for x in iris_data if x ]
    data_X = np.array(data_X,dtype=float)
    data_Y = [y[-1] for y in iris_data]
    for index,y in enumerate(data_Y):
        if y=='Iris-setosa':
            data_Y[index]=0
        elif y=='Iris-versicolor':
            data_Y[index] =1
        else:
            data_Y[index]=2
# 正规化数据集 X
def normalize(X, axis=-1, p=2):
    lp_norm = np.atleast_1d(np.linalg.norm(X, p, axis))
    lp_norm[lp_norm == 0] = 1
    return X / np.expand_dims(lp_norm, axis)


# 计算一个样本与数据集中所有样本的欧氏距离的平方
def euclidean_distance(one_sample, X):
    one_sample = one_sample.reshape(1, -1)
    X = X.reshape(X.shape[0], -1)
    distances = np.power(np.tile(one_sample, (X.shape[0], 1)) - X, 2).sum(axis=1)
    return distances
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
def plotK(x,scores):
    plt.plot(x,scores)
    plt.xlabel("value of max_iterations")
    plt.ylabel("adjusted_rand_score")
    plt.show()
class Kmeans():
    """Kmeans聚类算法.

    Parameters:
    -----------
    k: int
        聚类的数目.
    max_iterations: int
        最大迭代次数. 
    varepsilon: float
        判断是否收敛, 如果上一次的所有k个聚类中心与本次的所有k个聚类中心的差都小于varepsilon, 
        则说明算法已经收敛
    """
    def __init__(self, k=2, max_iterations=500, varepsilon=0.0001):
        self.k = k
        self.max_iterations = max_iterations
        self.varepsilon = varepsilon

    # 从所有样本中随机选取self.k样本作为初始的聚类中心
    def init_random_centroids(self, X):
        n_samples, n_features = X.shape
        centroids = np.zeros((self.k, n_features))
        for i in range(self.k):
            centroid = X[np.random.choice(range(n_samples))]
            centroids[i] = centroid
        return centroids

    # 返回距离该样本最近的一个中心索引[0, self.k)
    def _closest_centroid(self, sample, centroids):
        distances = euclidean_distance(sample, centroids)
        closest_i = np.argmin(distances)
        return closest_i

    # 将所有样本进行归类，归类规则就是将该样本归类到与其最近的中心
    def create_clusters(self, centroids, X):
        n_samples = np.shape(X)[0]
        clusters = [[] for _ in range(self.k)]
        for sample_i, sample in enumerate(X):
            centroid_i = self._closest_centroid(sample, centroids)
            clusters[centroid_i].append(sample_i)
        return clusters

    # 对中心进行更新
    def update_centroids(self, clusters, X):
        n_features = np.shape(X)[1]
        centroids = np.zeros((self.k, n_features))
        for i, cluster in enumerate(clusters):
            centroid = np.mean(X[cluster], axis=0)
            centroids[i] = centroid
        return centroids

    # 将所有样本进行归类，其所在的类别的索引就是其类别标签
    def get_cluster_labels(self, clusters, X):
        y_pred = np.zeros(np.shape(X)[0])
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i
        return y_pred

    # 对整个数据集X进行Kmeans聚类，返回其聚类的标签
    def predict(self, X):
        # 从所有样本中随机选取self.k样本作为初始的聚类中心
        centroids = self.init_random_centroids(X)
        labels = []
        # 迭代，直到算法收敛(上一次的聚类中心和这一次的聚类中心几乎重合)或者达到最大迭代次数
        for i in range(self.max_iterations):
            # print(i)
            # 将所有进行归类，归类规则就是将该样本归类到与其最近的中心
            clusters = self.create_clusters(centroids, X)
            former_centroids = centroids

            # 计算新的聚类中心
            centroids = self.update_centroids(clusters, X)

            # 如果聚类中心几乎没有变化，说明算法已经收敛，退出迭代
            diff = centroids - former_centroids
            if diff.any() < self.varepsilon:
                break
            # if i%100==0 and i!=0:
            #     print(i+"*")
            #     labels.append(self.get_cluster_labels(clusters,X))
        
        return self.get_cluster_labels(clusters, X)
        # return labels


def main():
    # 用Kmeans算法进行聚类
    clf = Kmeans(k=3)
    y_pred = clf.predict(data_X)
    print(len(y_pred))
    ari = adjusted_rand_score(data_Y,y_pred)
    print(ari)
    homogeneity_score, completeness_score, v_measure_score = homogeneity_completeness_v_measure(data_Y,y_pred)
    fmi_score = fowlkes_mallows_score(data_Y,y_pred)
    print(ari,homogeneity_score,completeness_score,v_measure_score)
    print(fmi_score)
    # iris_data k=3 0.7302382722834697 0.7514854021988338 0.7649861514489815 0.7581756800057784 0.8208080729114153
    clf = Kmeans(k=3,max_iterations=1000)
    y_pred = clf.predict(data_X)
    aris = []
    for i in range(1,6):
        clf = Kmeans(k=i)
        y_pred = clf.predict(data_X)
        ari = adjusted_rand_score(data_Y,y_pred)
        aris.append(ari)
    plotK(range(1,6),aris)
    # plotResult(data_X,y_pred,3) 
    
if __name__ == "__main__":
    main()