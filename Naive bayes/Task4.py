import numpy as np
import math
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.base import BaseEstimator
spambase = np.loadtxt('../data/spambase/spambase.data', delimiter = ",")
spamx = spambase[:, :57]
spamy = spambase[:, 57]
from sklearn.model_selection import train_test_split
trainX, testX, trainY, testY = train_test_split(spamx, spamy, test_size = 0.4, random_state = 32)
trainX.shape, trainY.shape, testX.shape, testY.shape
# YOUR CODE HERE
class myGaussianNB(BaseEstimator):
    def __init__(self):
        self.p_y = dict()
        self.p_x = dict()
    def fit (self,trainX,trainY):
        print("start training...")
        for y in trainY:
            if y not in self.p_y.keys():
                self.p_y[y]=1
            else:
                self.p_y[y]+=1
            
        for y in self.p_y:
            self.p_y[y]=self.p_y[y]/len(trainY)
        x_dict = dict()
        for y in self.p_y.keys():
            x_array= []
            for t_x,t_y in zip(trainX,trainY):
                if(t_y==y):
                    x_array.append(t_x)
            x_dict[y]=np.array(x_array)

        for x_key in x_dict:
            self.p_x[x_key]=[]
            for i in  range(len(x_dict[x_key][0])):
                means = np.mean(x_dict[x_key][:,i])
                var = np.var(x_dict[x_key][:,i])
                tup = (means,var)
                self.p_x[x_key].append(tup)
        pass
    def Gaussian(self,x,means,var):
        if var==0:
            var = 1e-6
        # exponent = math.exp(-((math.pow(x-means,2)/(2*var))))
        # return (1/math.sqrt(2*math.pi*var))*exponent

        return -0.5*math.log(2*math.pi*var)-0.5*math.pow((x-means),2)/var
    def predict(self,testX):
        print("start predict...")
        predictions =[]
        for row in testX:
            max_p=None
            index=None
            for y in self.p_y:
                p = math.log(self.p_y[y])
                # p=self.p_y[y]
                for i in range(len(row)):
                    # g_p= self.Gaussian(row[i], self.p_x[y][i][0], self.p_x[y][i][1])
                    p += (self.Gaussian(row[i], self.p_x[y][i][0], self.p_x[y][i][1]))
                if(max_p==None):
                    max_p = p
                    index = y
                elif(p>max_p):
                    max_p=p
                    index = y
            predictions.append(index)
        return predictions
        pass

def cross_validation(model,dataX,dataY,cv):
    accuracy=[]
    kf = KFold(10,shuffle=True)
    for train_index , test_index in kf.split(dataX):
        trainX = []
        trainY = []
        testX = []
        testY = []
        for index in train_index:
            trainX.append(dataX[index])
            trainY.append(dataY[index])
        for index in test_index:
            testX.append(dataX[index])
            testY.append(dataY[index])
        model.fit(trainX,trainY)
        predictions = model.predict(testX)
        accuracy.append(accuracy_score(testY,predictions))
        model.__init__()
    return accuracy
    pass
# test case
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

model = myGaussianNB()
predictions = cross_val_predict(model,spamx,spamy,cv=10)
accuracy = accuracy_score(predictions,spamy)
precision = precision_score(predictions,spamy)
recall = recall_score(predictions,spamy)
f1 = f1_score(predictions,spamy)
print("accuracy|precision|recall|f1")
print("%f|%f|%f|%f"%(accuracy,precision,recall,f1))


