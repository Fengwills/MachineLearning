import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler
# digits = load_digits()
# data_X = digits.data
# data_Y = digits.target
#
#
# mlp = MLPClassifier(random_state=42)
# predictions = cross_val_predict(mlp,data_X,data_Y,cv=10)
# accuracy = accuracy_score(data_Y,predictions)
# precision = precision_score(data_Y,predictions,average='weighted')
# recall = recall_score(data_Y,predictions,average='weighted')
# f1 = f1_score(data_Y,predictions,average='weighted')
# print("accuracy|precision|recall|f1")
# print("%f|%f|%f|%f"%(accuracy,precision,recall,f1))
def test():
    print("test")