import pandas as pd

# Location of dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Assign colum names to the dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
irisdata = pd.read_csv(url, names=names)  
print(irisdata.head())
# Assign data from first four columns to X variable
X = irisdata.iloc[:, 0:4]

# Assign data from first fifth columns to y variable
y = irisdata.select_dtypes(include=[object])  
print(y.head() )
print(y.Class.unique() )
from sklearn import preprocessing  
le = preprocessing.LabelEncoder()

y = y.apply(le.fit_transform)
print(y.Class.unique())
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test) 
from sklearn.neural_network import MLPClassifier  
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)  
mlp.fit(X_train, y_train.values.ravel())
predictions = mlp.predict(X_test)  
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
predictions = cross_val_predict(mlp,X,y,cv=10)
accuracy_micro = accuracy_score(y,predictions) 
accuracy_macro = accuracy_score(y,predictions)
precision_micro = precision_score(y,predictions,average='micro')
precision_macro = precision_score(y,predictions,average='macro')
recall_micro = recall_score(y,predictions,average='micro')
recall_macro = recall_score(y,predictions,average='macro')
f1_micro = f1_score(y,predictions,average='micro') 
f1_macro = f1_score(y,predictions,average='macro') 
print("accuracy|precision|recall|f1")
print("micro %f|%f|%f|%f"%(accuracy_micro,precision_micro,recall_micro,f1_micro))
print("macro %f|%f|%f|%f"%(accuracy_macro,precision_macro,recall_macro,f1_macro))

# print(confusion_matrix(y,predictions))  
# print(classification_report(y,predictions))
# print('accuracy'+str(accuracy))