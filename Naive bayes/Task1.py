import numpy as np
spambase = np.loadtxt('../data/spambase/spambase.data', delimiter = ",")
spamx = spambase[:, :57]
spamy = spambase[:, 57]
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

model = GaussianNB()

# YOUR CODE HERE
predictions = cross_val_predict(model,spamx,spamy,cv=10)
accuracy = accuracy_score(spamy,predictions)
precision = precision_score(spamy,predictions)
recall = recall_score(spamy,predictions)
f1 = f1_score(spamy,predictions)
print(accuracy,precision,recall,f1)