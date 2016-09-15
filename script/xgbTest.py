import pickle
import xgboost as xgb
import numpy as np
from sklearn.cross_validation import KFold, train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.grid_search import GridSearchCV
import pandas as pd
from sklearn.datasets import load_iris, load_digits, load_boston
from sklearn.cross_validation import  StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import numpy as np
from sklearn.feature_extraction import DictVectorizer

def encode_onehot(df, cols):
    """
    One-hot encoding is applied to columns specified in a pandas DataFrame.

    Modified from: https://gist.github.com/kljensen/5452382

    Details:

    http://en.wikipedia.org/wiki/One-hot
    http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with one-hot encoding
    """
    vec = DictVectorizer()

    vec_data = pd.DataFrame(vec.fit_transform(df[cols].to_dict(orient='records')).toarray())
    vec_data.columns = vec.get_feature_names()
    vec_data.index = df.index

    df = df.drop(cols, axis=1)
    df = df.join(vec_data)
    return df


def runInbuiltDataSet():
    iris = load_iris()
    Y = iris['target']
    X = iris['data']
    print X,type(Y)
    eval_size =0.10
    kf = StratifiedKFold(Y,10)
    train_indices,valid_indices = next(iter(kf))
    print train_indices,valid_indices


def splitDataset(x_train,y_train):
    eval_size =0.10
    kf = StratifiedKFold(y_train, 1. /eval_size)
    train_indices,valid_indices = next(iter(kf))
    X_train,Y_train = x_train.iloc[train_indices],y_train.iloc[train_indices]
    X_valid,Y_valid = x_train.iloc[valid_indices],y_train.iloc[valid_indices]
    return (X_train,Y_train),(X_valid,Y_valid)


def readFile():
    df = pd.read_csv("train.csv", header=0)
    x_train = df.iloc[: , :-1]
    y_train = df.iloc[:,-1]
    return x_train,y_train

def main():
    x_train,y_train = readFile()
    train_tuple,valid_tuple = splitDataset(x_train,y_train)
    train_data_x = train_tuple[0]
    train_data_y = train_tuple[1]

    cat_list = []
    print list(train_data_x.columns.values)
    for col in list(train_data_x.columns.values):
        #print train_data_x[col].dtypes
        if(train_data_x[col].dtypes == object):
            cat_list.append(col)

    lbl_enc = LabelEncoder()
    ohe = OneHotEncoder()
    #print cat_list
    ##test = pd.get_dummies(train_data_x[cat_list[0]])
    ##print test
    df = encode_onehot(train_data_x, cols=cat_list)
    print df.head() #train_data_x[item]
    #for item in cat_list:

        #train_data_x[item] = train_data_x[item].astype('category')
        #print type(train_data_x[item])
        #lbl_enc.fit(train_data_x[item].values)

        #train_data_x[item] = lbl_enc.transform(train_data_x[item].values)
        #print len(train_data_x[item]),item

        #train_data_x[item] = xtrain_cat

        #ohe.fit(train_data_x[item])
        #t = ohe.transform(train_data_x[item].values)













if __name__ == '__main__':
    main()