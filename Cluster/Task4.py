from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.datasets import load_iris
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import homogeneity_completeness_v_measure
import numpy as np
# iris = load_iris()
# iris_data = iris.data
# iris_target = iris.target
heart_data_path = '../data/heart/heart.dat'
with open(heart_data_path,'r') as f:
    heart_data = [line.strip().split(' ') for line in f.readlines()]
    data_X = [x[:-1] for x in heart_data ]
    # data_X = np.delete(data_X,[1,2,5,6,8,12],axis = 1)
    # print(data_X)
    data_X = np.array(data_X,dtype=float)
    data_Y = [y[-1] for y in heart_data]
    # for index,i in enumerate(data_Y):
    #     data_Y[index] = float(i)-1
    data_Y = np.array(data_Y)

model = GaussianMixture(n_components = 2)
# model.fit(iris_data)
model.fit(data_X)

# predictions = model.predict(iris_data)
predictions = model.predict(data_X)
# ari = adjusted_rand_score(iris_target,predictions)
ari = adjusted_rand_score(data_Y,predictions)
print(ari)
homogeneity_score, completeness_score, v_measure_score = homogeneity_completeness_v_measure(data_Y,predictions)
print(ari,homogeneity_score,completeness_score,v_measure_score)
#0.9038742317748124