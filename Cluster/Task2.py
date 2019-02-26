#K-Means
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
iris = load_iris()
iris_data = iris.data
iris_target = iris.target

model = KMeans(n_clusters = 3)
predictions = model.fit_predict(iris_data,iris_target)
ari = adjusted_rand_score(iris_target,predictions)
print(ari)
# 0.7302382722834697
