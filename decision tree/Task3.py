import numpy as np
import pandas as pd


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
import json

# read dataset
loans = pd.read_csv('../data/lendingclub/lending-club-data.csv', low_memory=False)
# Using safe_loan for target, and setting positive sample to 1, negative sample to -1, reomve bad_loan
loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
print(loans['safe_loans'])
del loans['bad_loans']
print(loans['safe_loans'])
features = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home_ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
           ]
target = 'safe_loans'
loans = loans[features + [target]]
#check the dataset
print(loans.head())

from sklearn.utils import shuffle
loans = shuffle(loans, random_state = 34)

split_line = int(len(loans) * 0.6)
train_data = loans.iloc[: split_line]
test_data = loans.iloc[split_line:]


# Using pd.get_dummies to obtain one-hot encoding
def one_hot_encoding(data, features_categorical):
    '''
    Parameter
    ----------
    data: pd.DataFrame

    features_categorical: list(str)
    '''

    for cat in features_categorical:
        one_encoding = pd.get_dummies(data[cat], prefix=cat)

        data = pd.concat([data, one_encoding], axis=1)

        del data[cat]

    return data
train_data = one_hot_encoding(train_data, features)
train_data.head()
#obtain feature name
one_hot_features = train_data.columns.tolist()
one_hot_features.remove(target)
print(one_hot_features)

#obtain one hot encoding for test data
test_data_tmp = one_hot_encoding(test_data, features)
test_data = pd.DataFrame(columns = train_data.columns)
for feature in train_data.columns:
    if feature in test_data_tmp.columns:
        test_data[feature] = test_data_tmp[feature].copy()
    else:
        test_data[feature] = np.zeros(test_data_tmp.shape[0], dtype = 'uint8')
print(test_data.head())
print(train_data.shape)
print(test_data.shape)

def information_entropy(labels_in_node):
    '''
    calculating information entropy for node

    Parameter
    ----------
    labels_in_node: np.ndarray, for example [-1, 1, -1, 1, 1]

    Returns
    ----------
    float: information entropy
    '''
    # number of samples
    num_of_samples = labels_in_node.shape[0]

    if num_of_samples == 0:
        return 0

    # number of positive samples
    num_of_positive = len(labels_in_node[labels_in_node == 1])

    # number of negative samples
    # YOUR CODE HERE
    num_of_negative = len(labels_in_node[labels_in_node==-1])

    # the probability of positive sample
    prob_positive = num_of_positive / num_of_samples

    # the probability of negative sample
    # YOUR CODE HERE
    prob_negative = num_of_negative / num_of_samples

    if prob_positive == 0:
        positive_part = 0
    else:
        positive_part = prob_positive * np.log2(prob_positive)

    if prob_negative == 0:
        negative_part = 0
    else:
        negative_part = prob_negative * np.log2(prob_negative)

    return - (positive_part + negative_part)


# compute information gains
def compute_information_gains(data, features, target, annotate=False):
    '''
    Parameter
    ----------
        data: pd.DataFrame，sample

        features: list(str)，feature name

        target: str, lable name

        annotate, boolean，print information gains for all features? False by default

    Returns
    ----------
        information_gains: dict, key: str, feature name
                                 value: float，information gains
    '''

    information_gains = dict()

    # Fore each feature, compute information gains
    for feature in features:

        left_split_target = data[data[feature] == 0][target]

        right_split_target = data[data[feature] == 1][target]

        # compute inforamtion entropy for left subtree
        left_entropy = information_entropy(left_split_target)

        # compute weight for left subtree
        left_weight = len(left_split_target) / (len(left_split_target) + len(right_split_target))

        # compute inforamtion entropy for right subtree
        # YOUR CODE HERE
        right_entropy = information_entropy(right_split_target)

        # compute weight for right subtree
        # YOUR CODE HERE
        right_weight = len(right_split_target) / (len(right_split_target) + len(left_split_target))

        # compute informatino entropy for data
        current_entropy = information_entropy(data[target])

        # compute information gains based on feature
        # YOUR CODE HERE
        gain = current_entropy - right_weight*right_entropy-left_weight*left_entropy

        # store information gains
        information_gains[feature] = gain

        if annotate:
            print(" ", feature, gain)

    return information_gains


def best_splitting_feature(data, features, target, criterion='information_gain', annotate=False):
    '''
    Parameters
    ----------
    data: pd.DataFrame,

    features: list(str)，

    target: str，

    criterion: str,

    annotate: boolean, default False，

    Returns
    ----------
    best_feature: str,

    '''
    if criterion == 'information_gain':
        if annotate:
            print('using information gain')

        # obtain the information gains
        information_gains = compute_information_gains(data, features, target, annotate)

        # Select the best feature according to the maximum information gains
        # YOUR CODE HERE
        best_feature = max(information_gains,key=information_gains.get)

        return best_feature,information_gains

    else:
        raise Exception("criterion is abnormal!", criterion)


# Obtain the number of sample for minority class
def intermediate_node_num_mistakes(labels_in_node):
    '''
    return the number of sample for minority class，For example [1, 1, -1, -1, 1]，return 2

    Parameter
    ----------
    labels_in_node: np.ndarray, pd.Series

    Returns
    ----------
    int：

    '''
    if len(labels_in_node) == 0:
        return 0
    # YOUR CODE HERE
    num_of_one = len(labels_in_node[labels_in_node == 1])
    # YOUR CODE HERE
    num_of_minus_one = len(labels_in_node[labels_in_node==-1])

    return num_of_one if num_of_minus_one > num_of_one else num_of_minus_one


def create_leaf(target_values):
    '''
    Compute the target value of corresponding leaf node, and store node information into dict


    Parameter:
    ----------
    target_values: pd.Series,

    Returns:
    ----------
    leaf: dict，leaf node，
            leaf['splitting_features'], None，No need to split feature
            leaf['left'], None，leaf node do not have left subtree
            leaf['right'], None，leaf node do not have right subtree
            leaf['is_leaf'], True, is leaf node?
            leaf['prediction'], int, the predicton of leaf node
    '''
    # create leaf node
    leaf = {'splitting_feature': None,
            'left': None,
            'right': None,
            'is_leaf': True}

    # obtain the numbe of positive samples and negative samples
    num_ones = len(target_values[target_values == +1])
    num_minus_ones = len(target_values[target_values == -1])

    if num_ones > num_minus_ones:
        leaf['prediction'] = 1
    else:
        leaf['prediction'] = -1

    # 返回叶子结点
    return leaf


def decision_tree_create(data, features, target, criterion='information_gain', current_depth=0, max_depth=10,
                         annotate=False):
    '''
    Parameter:
    ----------
    data: pd.DataFrame,

    features: iterable,

    target: str,

    criterion: 'str', 'information_gain'

    current_depth: int,

    max_depth: int,

    Returns:
    ----------
    dict, dict['is_leaf']          : False,
          dict['prediction']       : None,
          dict['splitting_feature']: splitting_feature,
          dict['left']             : dict
          dict['right']            : dict
    '''

    if criterion not in ['information_gain', 'gain_ratio', 'gini']:
        raise Exception("criterion is abnormal!", criterion)

    remaining_features = features[:]

    target_values = data[target]
    print("-" * 50)
    print("Subtree, depth = %s (%s data points)." % (current_depth, len(target_values)))
    if(len(target_values)<50):
        print("prepruning")
        return create_leaf(target_values)
    # Stop condition 1
    # all of the samples have the same lable
    # Using function intermediate_node_num_mistakes
    # YOUR CODE HERE
    if len(set(target_values))==1:
        print("Stopping condition 1 reached.")
        return create_leaf(target_values)

        # Stop condition 2
    # all features have been used，remaining_features is null
    # YOUR CODE HERE
    if not remaining_features:
        print("Stopping condition 2 reached.")
        return create_leaf(target_values)

        # Stop condition 3
    # the depth of node is maximum depth
    # YOUR CODE HERE
    if  current_depth==max_depth:
        print("Reached maximum depth. Stopping for now.")
        return create_leaf(target_values)

        # obtain the optimal feature for splitting
    # Using function best_splitting_feature
    # YOUR CODE HERE
    splitting_feature,ig = best_splitting_feature(data,remaining_features,target)

    # split data
    # left subtree
    left_split = data[data[splitting_feature] == 0]

    # right subtree
    # YOUR CODE HERE
    right_split = data[data[splitting_feature] == 1]

    # remove optimal feature from ramaining features
    remaining_features.remove(splitting_feature)

    # print splitting feature and the number of samples of left tree and right tree
    print("Split on feature %s. (%s, %s) information_gain %s" % ( \
        splitting_feature, len(left_split), len(right_split),ig))

    # If all of the samples are splited into one subtree, create left node
    # left tree
    if len(left_split) == len(data):
        print("Creating leaf node.")
        return create_leaf(left_split[target])

    # right tree
    if len(right_split) == len(data):
        print("Creating right node.")
        # YOUR CODE HERE
        return  create_leaf(right_split[target])

    # create left tree recursively
    left_tree = decision_tree_create(left_split, remaining_features, target, criterion, current_depth + 1, max_depth,
                                     annotate)

    # create right tree recursively
    # YOUR CODE HERE
    right_tree = decision_tree_create(right_split, remaining_features, target, criterion, current_depth + 1, max_depth,
                                      annotate)

    # return nodes except left nodes
    return {'is_leaf': False,
            'prediction': None,
            'splitting_feature': splitting_feature,
            'left': left_tree,
            'right': right_tree}


def classify(tree, x, annotate=False):
    '''

    Parameters
    ----------
    tree: dict

    x: pd.Series，sample to be predicted

    annotate： boolean,

    Returns
    ----------
    return the predicted lable
    '''
    if tree['is_leaf']:
        if annotate:
            print("At leaf, predicting %s" % tree['prediction'])
        return tree['prediction']
    else:
        split_feature_value = x[tree['splitting_feature']]
        if annotate:
            print("Split on %s = %s" % (tree['splitting_feature'], split_feature_value))
        if split_feature_value == 0:
            return classify(tree['left'], x, annotate)
        else:
            return classify(tree['right'], x, annotate)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict
from pandas import DataFrame
import matplotlib.pyplot  as plt
def predict(tree, data):
    '''

    Parameter
    ----------
    tree: dict, decision tree model

    data: pd.DataFrame, data

    Returns
    ----------
    predictions：np.ndarray, the predicted lables
    '''
    print("start predict")
    predictions = np.zeros(len(data))

    # YOUR CODE HERE
    for i in range(0,len(data)):
        predictions[i] = classify(tree, data.iloc[i])

    return predictions

def cross_validation(data,maxdepth,cv,target):
    length = len(data)
    l = len(data)//cv
    x_folds = []
    for i in range(cv):
        if(i!=9):
            start = i*l
            end = (i+1)*l
            x_folds.append(data[start:end])
        else:
            x_folds.append(data[i*l:len(data)])
    # y_folds = list(np.hsplit(datay,cv))
    predictions=[]
    for i in range(cv):
        X_train = np.vstack(x_folds[:i]+x_folds[i+1:])

        X_val = x_folds[i]
        features = X_val.columns.tolist()
        X_train = DataFrame(X_train, columns=features)
        features.remove(target)
        X_train = one_hot_encoding(X_train, features)
        one_hot_features = X_train.columns.tolist()
        one_hot_features.remove(target)
        my_decision_tree = decision_tree_create(X_train, one_hot_features, target, current_depth=0, max_depth=maxdepth,
                                                annotate=False)
        X_val_tmp = one_hot_encoding(X_val,features)
        X_val = pd.DataFrame(columns=X_train.columns)
        for feature in X_train.columns:
            if feature in X_val_tmp.columns:
                X_val[feature] = X_val_tmp[feature].copy()
            else:
                X_val[feature] = np.zeros(X_val_tmp.shape[0], dtype='uint8')

        p = predict(my_decision_tree,X_val)
        predictions=predictions+p.tolist()
    return predictions
predictions = cross_validation(loans,6,10,target)
accuracy = accuracy_score(loans[target],predictions)
precision = precision_score(loans[target],predictions)
recall = recall_score(loans[target],predictions)
f1 = f1_score(loans[target],predictions)
print(accuracy,precision,recall,f1)
scores = []
for i in range(1,11):
    print("*"*50)
    print("max_depth= %s"%i)
    predictions = cross_validation(loans,i,10,target)
    accuracy = accuracy_score(loans[target],predictions)
    scores.append(accuracy)
print("start paint")
plt.plot(range(1,11),scores)
plt.xlabel("value of maximum depth")
plt.ylabel("accuracy")
plt.savefig("Task3.png")
plt.show()