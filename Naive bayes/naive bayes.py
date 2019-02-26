import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_predict
spambase = np.loadtxt('../data/spambase/spambase.data', delimiter = ",")
spamx = spambase[:, :57]
spamy = spambase[:, 57]
model = GaussianNB()

model.fit(spamx,spamy)
# YOUR CODE HERE

prediction = cross_val_predict(model, spamx, spamy, cv = 10)
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

print(accuracy_score(spamy, prediction))
print(precision_score(spamy, prediction))
print(recall_score(spamy, prediction))
print(f1_score(spamy, prediction))