from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import numpy as np
spam_data_path = '../data/spambase/spambase.data'

with open(spam_data_path,'r') as f :
    spam_data = [line.strip().split(',') for line in f.readlines()]
    data_X = [x[:-1] for x in spam_data ]
    data_X = np.array(data_X,dtype=float)
    data_Y = [y[-1] for y in spam_data]
    data_Y = np.array(data_Y,dtype=float)

model = LogisticRegression()
predictions = cross_val_predict(model,data_X,data_Y,cv=10)
accuracy = accuracy_score(predictions,data_Y)
precision = precision_score(predictions,data_Y)
recall = recall_score(predictions,data_Y)
f1 = f1_score(predictions,data_Y)
print("accuracy|precision|recall|f1")
print("%f|%f|%f|%f"%(accuracy,precision,recall,f1))
