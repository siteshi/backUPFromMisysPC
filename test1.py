#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pickle
import xgboost as xgb
import numpy as np
from sklearn.covariance.tests.test_covariance import X_1d
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
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.feature_extraction.text import  TfidfTransformer
from sklearn.feature_extraction import text

d = {'a':1,'b':2}
for i in d:
    print i


df = pd.read_csv("train.csv", header=0)
df[['Product_Info_3']] = df[['Product_Info_3']].astype(object)
x_train = df.iloc[: , :-1]
y_train = df.iloc[:,-1]
#df = encode_onehot(x_train, cols=["Product_Info_2"])
#print df.dtypes #train_data_x[item]

ctv = CountVectorizer()
stop_words = text.ENGLISH_STOP_WORDS.union(['the','on'])
ctv.__init__(stop_words = stop_words)

text_data = list(x_train.apply(lambda x: '%s%s' %(x['Product_Info_2'],x['Product_Info_3']),axis =1))
print text_data
trans_text_data = ctv.fit_transform(text_data)
print ctv.get_feature_names()
print trans_text_data.todense()


docs_new = ['God is love and fast', 'OpenGL on the GPU is fast','GPU is created by god and we all love god and gpu']
X_new_counts = ctv.fit_transform(docs_new)
print ctv.vocabulary_
print ctv.get_feature_names()
print X_new_counts.todense()

tfidf_transformer = TfidfTransformer()
tf_transformer = TfidfTransformer().fit(X_new_counts)
X_train_tfidf = tf_transformer.fit_transform(X_new_counts)
print ctv.get_feature_names()
print X_train_tfidf.todense()
#print

