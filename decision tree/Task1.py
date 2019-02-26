import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

dota2results = np.loadtxt('../data/dota2Dataset/dota2Train.csv', delimiter=',')
dota2x = dota2results[:, 1:]
dota2y = dota2results[:, 0]
model = DecisionTreeClassifier(max_depth = 10)
# YOUR CODE HERE
predicted =cross_val_predict(model,dota2x,y=dota2y,cv=10)
accuracy=accuracy_score(dota2y,predicted)
precision_score(dota2y,predicted)
recall=recall_score(dota2y,predicted)
f1=f1_score(dota2y,predicted)
print(accuracy,recall,f1)
scores=[]
for i in range(1,11):
    model = DecisionTreeClassifier(max_depth=i)
    predicted = cross_val_predict(model,dota2x,y=dota2y,cv=10)
    accuracy = accuracy_score(dota2y,predicted)
    scores.append(accuracy)
plt.plot(range(1,11),scores)
plt.xlabel("value of maximum depth")
plt.ylabel("accuracy")
plt.savefig("Task1.png")
plt.show()
