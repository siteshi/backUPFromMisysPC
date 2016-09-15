#this util lib is for machine learning task
#it has some standard function which can be dircetly plug and play
#Author: Sitesh Indra
#Email: sitesh.indra@misys.com
#Date: 29/07/2016

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


def giveTrainDataFrameFromFile(fileName,columnNameOfResponse):
    df = pd.read_csv("train.csv", header=0)
    Y_train = df[columnNameOfResponse]
    df.drop(columnNameOfResponse,axis = 1)
    X_train = df
    return X_train,Y_train

def hotEncoding(dataFrame,colName,choice):

    if(choice == "panda"):
        dummyColumns = pd.get_dummies(dataFrame[colName])

    elif (choice == "dict"):
        vec = DictVectorizer()
        vec_data = pd.DataFrame(vec.fit_transform(dataFrame[colName].to_dict(orient='records')).toarray())
        vec_data.columns = vec.get_feature_names()
        vec_data.index = dataFrame.index
        dummyColumns = vec_data
    else:
        exit(-1)

    dataFrame = dataFrame.drop(colName, axis=1)
    dataFrame = dataFrame.join(dummyColumns)
    return dataFrame







def splitDataset(x_train,y_train,eval_size):
    kf = StratifiedKFold(y_train, 1. /eval_size)
    train_indices,valid_indices = next(iter(kf))
    X_train,Y_train = x_train.iloc[train_indices],y_train.iloc[train_indices]
    X_valid,Y_valid = x_train.iloc[valid_indices],y_train.iloc[valid_indices]
    return (X_train,Y_train),(X_valid,Y_valid)


def giveProcessedText():
    return True






if __name__ == "__main__":
    X_train,Y_train = giveTrainDataFrameFromFile("train.csv","Response")
    print type(Y_train)
    #print type(x["Product_Info_2"]), type(x[["Product_Info_2"]])
    X_train_with_dummy = hotEncoding(X_train,["Product_Info_2"],"panda")
    a,b = splitDataset(X_train_with_dummy,Y_train,0.10)
    print a, b




