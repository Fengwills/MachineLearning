from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler
import random
import numpy as np

class MyLogisiticRegression(BaseEstimator):
    def __init__(self):
        self.iters=1000
        self.alpha = 0.2
        self.theta = None
        self.b = None
        # l2 regularization hyper parameter
        self.C = 1.0
    def sigmoid(self,x):
        return 1.0 / (1.0 + np.exp(-x))
    def fit(self,datax,datay):
        # data size
        len_y = datay.shape[0]
        self.theta = np.random.randn(datax.shape[1],1)
        self.b = 0.0
        #gradient descent
        for i in range(self.iters):
            wx = np.dot(datax,self.theta)
            h = self.sigmoid(wx+self.b)
            test = np.dot(np.transpose(datay),np.log(h))
            LL = test + np.dot(np.transpose(1-datay),np.log(1e-10+1-h))
            #loss with l2
            cost = -1.0/len_y*LL+self.C/(2*len_y)*np.dot(np.transpose(self.theta),self.theta)
            gradient_w = 1.0/len_y*np.dot(np.transpose(datax),h-datay)+self.C/len_y*self.theta
            gradient_b = 1.0/len_y*np.sum(h-datay)
            self.theta -=self.alpha*gradient_w
            self.b -=self.alpha*gradient_b
            print("iter = %d cost=%f"%(i,cost))
        pass
    def predict(self,testx):
        predictions = np.dot(testx,self.theta)+self.b
        for i,p in enumerate(predictions):
            if(p>=0.5):
                predictions[i]=1
            else:
                predictions[i]=0
        return predictions
        pass

spam_data_path = '../data/spambase/spambase.data'
with open(spam_data_path,'r') as f:
    spam_data = [line.strip().split(',') for line in f.readlines()]
    random.shuffle(spam_data)
    data_X = [x[:-1] for x in spam_data]
    data_X = np.array(data_X,dtype=float)
    data_Y = [y[-1] for y in spam_data]
    data_Y = np.array(data_Y,dtype=float)
    data_Y = data_Y[:,np.newaxis]
    sc = StandardScaler()
    data_X = sc.fit_transform(data_X)

myLogisiticRegression = MyLogisiticRegression()
predictions = cross_val_predict(myLogisiticRegression,data_X,data_Y,cv=10)
accuracy = accuracy_score(data_Y,predictions)
precision = precision_score(data_Y,predictions)
recall = recall_score(data_Y,predictions)
f1 = f1_score(data_Y,predictions)
print("accuracy|precision|recall|f1")
print("%f|%f|%f|%f"%(accuracy,precision,recall,f1))
# accuracy|precision|recall|f1
# 0.913280|0.835632|0.937500|0.883640