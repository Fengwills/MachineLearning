from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
#load dataset
dataset = load_iris()
data = dataset.data
target = dataset.target
# standardScaler
sc = StandardScaler()
sc.fit_transform(data)
model = SVC()
predictions = cross_val_predict(model,data,target,cv=10)
accuracy = accuracy_score(predictions,target)
precision = precision_score(predictions,target,average='weighted')
recall = recall_score(predictions,target,average='weighted')
f1 = f1_score(predictions,target,average='weighted')
print("accuracy|precision|recall|f1")
print("%f|%f|%f|%f"%(accuracy,precision,recall,f1))
# accuracy|precision|recall|f1
# 0.980000|0.980133|0.980000|0.980002