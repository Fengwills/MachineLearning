from sklearn.cluster import SpectralClustering

from sklearn.metrics import adjusted_rand_score
from sklearn.datasets import load_iris
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
iris = load_iris()
iris_data = iris.data
Normalizer().fit_transform(iris_data)
iris_target = iris.target
# metrics_metrixs = (-1* metrics.pairwise.pairwise_distances(iris_data))
# metrics_metrixs +=-1*metrics_metrixs.min()
# model = Spectralclustring(metrics_metrixs,n_clusters = 3)
model = SpectralClustering(n_clusters=3)
predictions = model.fit_predict(iris_data,iris_target)
ari = adjusted_rand_score(iris_target,predictions)
print(ari)
# 0.7436826319432357