import pandas as pd

# Location of dataset
# url = " http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat"
url = '../data/heart/heart.dat'
# Assign colum names to the dataset
names = ['age','sex,']

# Read dataset to pandas dataframe
heartdata = pd.read_csv(url,sep=' ')  
print(heartdata.head())
# Assign data from first four columns to X variable
X = heartdata.iloc[:, :-1]

# Assign data from first fifth columns to y variable
# y = heartdata.select_dtypes(include=[object]) 
y  = heartdata.iloc[:,-1:]
print(y.head() )
# print(y.Class.unique() )
from sklearn import preprocessing  
le = preprocessing.LabelEncoder()

y = y.apply(le.fit_transform)
# print(y.Class.unique())
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()
scaler.fit(X)  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test) 
from sklearn.neural_network import MLPClassifier  
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=10000)  
mlp.fit(X_train, y_train.values.ravel())
predictions = mlp.predict(X_test)  
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))
from sklearn.metrics import accuracy_score
# accuracy = accuracy_score(y_test,predictions)  
# print('accuracy'+str(accuracy))
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
predictions = cross_val_predict(mlp,X,y,cv=10)
accuracy_macro = accuracy_score(y,predictions)
precision_macro = precision_score(y,predictions,average='macro')
recall_macro = recall_score(y,predictions,average='macro')
f1_macro = f1_score(y,predictions,average='macro') 
print("accuracy|precision|recall|f1")
print("macro %f|%f|%f|%f"%(accuracy_macro,precision_macro,recall_macro,f1_macro))
#  70.0  1.0  4.0  130.0  322.0  0.0  2.0  109.0  0.0.1  2.4  2.0.1  3.0  \
# 0  67.0  0.0  3.0  115.0  564.0  0.0  2.0  160.0    0.0  1.6    2.0  0.0
# 1  57.0  1.0  2.0  124.0  261.0  0.0  0.0  141.0    0.0  0.3    1.0  0.0
# 2  64.0  1.0  4.0  128.0  263.0  0.0  0.0  105.0    1.0  0.2    2.0  1.0
# 3  74.0  0.0  2.0  120.0  269.0  0.0  2.0  121.0    1.0  0.2    1.0  1.0
# 4  65.0  1.0  4.0  120.0  177.0  0.0  0.0  140.0    0.0  0.4    1.0  0.0

#    3.0.1  2
# 0    7.0  1
# 1    7.0  2
# 2    7.0  1
# 3    3.0  1
# 4    7.0  1
#    2
# 0  1
# 1  2
# 2  1
# 3  1
# 4  1
# [[18  6]
#  [ 9 21]]
#              precision    recall  f1-score   support

#           0       0.67      0.75      0.71        24
#           1       0.78      0.70      0.74        30

# avg / total       0.73      0.72      0.72        54