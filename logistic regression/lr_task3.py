from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import BaseEstimator
import numpy as np
from sklearn.preprocessing import StandardScaler

spam_data_path = '../data/spambase/spambase.data'
with open(spam_data_path, 'r')as f:
    spam_data = [x.strip().split(',') for x in f.readlines()]
    np.random.shuffle(spam_data)
    for cnt, i in enumerate(spam_data):
        spam_data[cnt] = [float(j) for j in spam_data[cnt]]
    x_data = np.array([x[:-1] for x in spam_data])
    x_data = StandardScaler().fit_transform(x_data)
    y_data = np.array([x[-1] for x in spam_data])


# implement my Logistic Regression
class MyLogisticRegression(BaseEstimator):
    def __init__(self):
        self.weight = None
        self.bias = None
        self.lr = 0.2
        self.iter = 1000
        # l2 regularization hyper parameter
        self.l2_lambd = 1

    def gradient_descent(self, train_x, train_y):
        # data size
        sample_size, feature_size = train_x.shape

        # predict using weight
        predict_y = self.compute_g0(train_x)

        # y_diff = (1 - predict_y) - train_y
        y_diff = predict_y - train_y

        # compute and record loss
        one_case = np.multiply(-train_y, np.log(predict_y))
        # using a small number 1e-10 to avoid NAN
        zero_case = np.multiply(-(1 - train_y), np.log(1e-10 + 1 - predict_y))
        # compute l2 loss
        l2_loss = self.l2_regular_cost()
        loss = (np.sum(one_case + zero_case) + l2_loss) / (2 * sample_size)
        print('loss:', loss)

        # compute gradient
        gradient = (np.dot(train_x.T, y_diff) + self.l2_lambd * self.weight) / sample_size
        self.weight = self.weight - self.lr * gradient
        return loss

    def fit(self, train_x, train_y):
        # init w and b based on data
        sample_size, feature_size = train_x.shape
        print('training data size:', train_x.shape)
        self.weight = np.random.normal(size=feature_size)

        iter_cnt = 0
        print('start training')
        losses = list()
        while iter_cnt < self.iter:
            # while iter_cnt < self.iter and np.any(np.absolute(self.weight) > 1e-5):
            if (iter_cnt + 1) % 100 == 0:
                print('iter ', iter_cnt + 1)
            loss = self.gradient_descent(train_x, train_y)
            losses.append(loss)
            iter_cnt += 1
        return losses

    def predict(self, test_x):
        return self.compute_g0(test_x) >= 0.5

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def compute_g0(self, x):
        return self.sigmoid(np.dot(x, self.weight))

    def l2_regular_cost(self):
        return self.l2_lambd * np.sqrt(np.sum(np.square(self.weight)))


model = MyLogisticRegression()

predicted = cross_val_predict(model, x_data, y_data, cv=10)

# plot figure
# import matplotlib.pyplot as plt
#
# losses = model.fit(x_data, y_data)
# plt.plot(losses)
# plt.title('Iteration vs Loss')
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
# plt.show()

accuracy = accuracy_score(y_data, predicted)
precision = precision_score(y_data, predicted)
recall = recall_score(y_data, predicted)
f1 = f1_score(y_data, predicted)

print('Accuracy | Precision | Recall | F1')
print(accuracy, '|', precision, '|', recall, "|", f1)

# Accuracy | Precision | Recall | F1
# 0.9191480113018909 | 0.9063733784545968 | 0.8863761720904578 | 0.8962632459564974

# while add l2 did not improve the accuracy, and loss image is same as task2
