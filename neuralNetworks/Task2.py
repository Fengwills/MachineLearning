from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd


#cost function
class QuadraticCost(object):
    def fn(a,y):
        return 0.5*np.linalg.norm(a-y)**2
    def delta(a,y):
        return (a-y)
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))
def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

#MLP estimator
class myMultilayerPerceptron(BaseEstimator):
    def __init__(self,sizes,mini_batch_size,loss = QuadraticCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.loss = loss
        self.weight_initializer()
        self.loss = loss
        self.mini_batch_size = mini_batch_size
        self.eta = 0.02
        self.epochs = 60

    def weight_initializer(self):
        self.biases = [np.random.randn(1,y) for y in self.sizes[1:]]
        self.weight = [np.random.randn(x,y) for y,x in
                       zip(self.sizes[1:],self.sizes[:-1])]
    def feed_forward(self,a):
        for w,b in zip(self.weight,self.biases):
            a = sigmoid(np.dot(a,w)+b)
        return a
    def backprop(self,x,y):
        nabla_b = [np.zeros(b.shape) for b in self.biases ]
        nabla_W = [np.zeros(W.shape) for W in self.weight]
        activation = x
        #every layer activations
        activations = [x]
        #every layer z vector
        zs = []
        for b,w in zip(self.biases,self.weight):
            z = np.dot(activation,w)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        #last layer error
        delta = self.loss.delta(activations[-1],y)
        #last layer dC/db
        nabla_b[-1] = delta*sigmoid_prime(zs[-1])
        #last layer dC/dW
        nabla_W[-1] = np.dot(np.transpose(activations[-2]),delta*sigmoid_prime(zs[-1]))

        for l in range(2,self.num_layers):
            sp = sigmoid_prime(zs[-l])
            delta = np.dot(delta,np.transpose(self.weight[-l+1]))*sp
            nabla_b[-l] = delta
            nabla_W[-l] = np.dot(np.transpose(activations[-l-1]),delta)
        return nabla_b,nabla_W

    def SGD(self,training_data_X,training_data_Y,epochs,mini_batch_size,eta,test_data=None):
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
        # training_data_X = list(training_data_X)
        n = len(training_data_X)
        for i in range(epochs):
            mini_batches_X = [
                training_data_X[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            mini_batches_Y = [
                training_data_Y[k:k + mini_batch_size]
                for k in range(0,n,mini_batch_size)]

            for mini_batch_X,mini_batch_Y in zip(mini_batches_X,mini_batches_Y):
                self.update_mini_batch(mini_batch_X,mini_batch_Y, eta)
            if test_data:
                print ("Epoch {0}: {1} / {2}".format(
                    i, self.evaluate(test_data), n_test))
            else:
                print ("Epoch {0} complete".format(i))
        pass
    #update b,W based on bacth samples
    def update_mini_batch(self,mini_batch_X,mini_batch_Y,eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_W = [np.zeros(W.shape) for W in self.weight]
        for x,y in zip(mini_batch_X,mini_batch_Y):
            x = x.reshape(1,len(x))
            delta_b,delta_W = self.backprop(x,y)
            nabla_b = [b+db for b,db in zip(nabla_b,delta_b)]
            nabla_W = [w+dw for w,dw in zip(nabla_W,delta_W)]
        self.biases = [b-(eta/len(mini_batch_X)*nb) for b,nb in zip(self.biases,nabla_b)]
        self.weight = [w-(eta/len(mini_batch_X)*nw) for w,nw in zip(self.weight,nabla_W)]

    def fit(self,data_X,data_Y):
        data_Y = onehot_encoding(data_Y)
        self.SGD(data_X,data_Y,self.epochs,self.mini_batch_size,self.eta)
    def predict(self,test_X):
        predictions = [np.argmax(self.feed_forward(x)) for x in test_X]
        return predictions
def onehot_encoding(data):
    onehot = []
    for num in data:
        one = np.zeros(10)
        one[num] = 1
        onehot.append(one)

    return np.array(onehot)
kaggle_mnist = pd.read_csv('../data/kaggle_mnist/mnist_train.csv')
kaggle_mnist = kaggle_mnist.values
data_Y = kaggle_mnist[:,0]

data_X = kaggle_mnist[:,1:]
myMLP = myMultilayerPerceptron([784,30,10],20)
train_x,test_x,train_y,test_y = train_test_split(data_X,data_Y,test_size=0.3,random_state=0)

# train_y = onehot_encoding(train_y)

myMLP.fit(train_x,train_y)
predictions = myMLP.predict(test_x)
# predictions = cross_val_predict(myMLP,data_X,data_Y,cv=10)
accuracy = accuracy_score(test_y,predictions)
precision = precision_score(test_y,predictions,average='weighted')
recall = recall_score(test_y,predictions,average='weighted')
f1 = f1_score(test_y,predictions,average='weighted')
# accuracy = accuracy_score(data_Y,predictions)
# precision = precision_score(data_Y,predictions,average='weighted')
# recall = recall_score(data_Y,predictions,average='weighted')
# f1 = f1_score(data_Y,predictions,average='weighted')
print("accuracy|precision|recall|f1")
print("%f|%f|%f|%f"%(accuracy,precision,recall,f1))

# accuracy|precision|recall|f1
# 0.850397|0.850904|0.850397|0.849655
# accuracy|precision|recall|f1
# 0.844714|0.844344|0.844714|0.844169