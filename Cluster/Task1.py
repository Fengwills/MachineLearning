#from __future__ import absolute_import
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.datasets import load_iris
from sklearn.preprocessing import Normalizer
from neuralNetworks.Task1 import test
from sklearn.model_selection import cross_val_predict
# iris = load_iris()
# iris_data = iris.data
# Normalizer().fit_transform(iris_data)
# iris_target = iris.target
# model = AgglomerativeClustering(n_clusters = 3,linkage = 'average')
# predictions = model.fit_predict(iris_data)
# ari = adjusted_rand_score(iris_target,predictions)
# print(ari)
#0.7591987071071522
test()