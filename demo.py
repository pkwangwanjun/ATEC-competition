# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

import sys
sys.path.append('/Users/wanjun/Desktop/LightGBM/python-package')
import lightgbm as lgb





#994731
#print(len(data))
#print(len(np.unique(data.index)))
 #0    977884
 #1     12122
#-1      4725
#pd.value_counts(data['label'])



def lgbmodel(x,y):
    #train_data = lgb.Dataset(data, label=label)
    lgbc=lgb.LGBMClassifier(n_estimators=500,max_depth=6,num_leaves=65,class_weight={0:1,1:1.41},learning_rate=0.01,subsample=0.8,random_state=0,n_jobs=-1,objective='binary')
    scores=cross_validate(estimator=lgbc,X=x,y=y,cv=10,scoring=make_scorer(roc_auc_score),n_jobs=-1,verbose=-1)
    print(scores['test_score'].mean())
    return scores,lgbc

def select_feature():
    pass

if __name__=='__main__':
    data=pd.read_csv('/Users/wanjun/Desktop/蚂蚁金服比赛/atec_anti_fraud_train.csv',index_col=0)
    data_clean=data[data['label']!=-1]
    y=data_clean['label']
    x=data_clean[data_clean.columns[data.columns!='label']]
