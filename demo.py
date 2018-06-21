# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
#from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
#import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

import sys
import lightgbm as lgb

import gc

#import mlxtend
from ycimpute.imputer import iterforest
from ycimpute.imputer import simple
from sklearn.metrics import confusion_matrix,roc_curve

from bayes_opt import BayesianOptimization
from sklearn.model_selection import StratifiedKFold

import warnings
warnings.filterwarnings("ignore")


#994731
#print(len(data))
#print(len(np.unique(data.index)))
 #0    977884
 #1     12122
#-1      4725
#pd.value_counts(data['label'])



def score_atc(y_true,y_pred_prob):

    fpr, tpr, thresholds = roc_curve(y_true,y_pred_prob, pos_label=1)

    score=0.4*tpr[(fpr>=0.001)][0]+0.3*tpr[(fpr>=0.005)][0]+0.3*tpr[(fpr>=0.01)][0]

    return score

def train_test(x,y):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)
    lgbc=lgb.LGBMClassifier(n_estimators=500,max_depth=-1,num_leaves=65,learning_rate=0.01,subsample=0.8,sub_feature=0.8,random_state=0,n_jobs=2,objective='binary',is_unbalance=True)
    lgbc.fit(x_train,y_train)
    pred=lgbc.predict(x_test)
    pred_prob=lgbc.predict_proba(x_test)
    print(precision_score(y_test,pred,pos_label=1))
    print(recall_score(y_test,pred,pos_label=1))
    print(roc_auc_score(y_test,pred_prob[:,1]))
    return y_test,pred_prob

def lgbmodel(x,y):
    #train_data = lgb.Dataset(data, label=label)
    #class_weight={0:1,1:1.41}
    lgbc=lgb.LGBMClassifier(n_estimators=500,max_depth=-1,num_leaves=80,learning_rate=0.01,subsample=0.8,random_state=0,n_jobs=4,objective='binary',is_unbalance=True)
    scores=cross_validate(estimator=lgbc,X=x,y=y,cv=10,scoring=make_scorer(roc_auc_score),n_jobs=1,verbose=-1)
    print(scores['test_score'].mean())
    return scores


def lgbccv(n_estimators,num_leaves,min_child_samples,reg_alpha,reg_lambda,subsample,colsample_bytree):
    skf=StratifiedKFold(n_splits=5)
    skf.get_n_splits(data,label)
    score=[]
    for train_index, test_index in skf.split(data,label):
        lgbc=lgb.LGBMClassifier(n_estimators=int(n_estimators),max_depth=-1,num_leaves=int(num_leaves),
                                learning_rate=0.01,subsample=max(min(subsample, 1), 0),
                                colsample_bytree=max(min(colsample_bytree, 1), 0),random_state=666,
                                min_child_samples=int(min_child_samples),
                                reg_alpha=max(reg_alpha,0),reg_lambda=max(reg_lambda,0),
                                n_jobs=4,objective='binary',
                                is_unbalance=True)
        lgbc.fit(data[train_index],label[train_index])
        pred_prob=lgbc.predict_proba(data[test_index])
        score.append(score_atc(label[test_index],pred_prob[:,1]))
    return np.array(score).mean()


def rfmodel(x,y):
    pass

def select_feature():
    pass

def sub_pred():
    data=pd.read_csv('atec_anti_fraud_train.csv',index_col=0)
    data=data[data['label']!=-1]
    y=data['label']
    x=data[data.columns[data.columns!='label']]
    gc.collect()
    lgbc=lgb.LGBMClassifier(n_estimators=500,max_depth=-1,num_leaves=100,learning_rate=0.01,subsample=0.8,sub_feature=0.8,random_state=0,n_jobs=5,objective='binary',is_unbalance=True)
    lgbc.fit(x,y)
    test=pd.read_csv('atec_anti_fraud_test_a.csv',index_col=0)
    pred_prob=lgbc.predict_proba(test)
    temp=pd.DataFrame(pred_prob[:,1],index=test.index)
    temp.reset_index(inplace=True)
    temp.columns=['id','score']
    temp.to_csv('test_wtf.csv',index=None)


def compelet_data():
    data=pd.read_csv('./data/atec_anti_fraud_train.csv',index_col=0)
    y=data['label']
    y[y==-1]=1
    x=data[data.columns[data.columns!='label']]
    gc.collect()
    print('load data')
    print('counting')
    #x_complete=iterforest.IterImput().complete(x.values)
    x_complete=simple.SimpleFill(fill_method='mean').complete(x.values)
    print('count out')
    x_complete=pd.DataFrame(x_complete)
    print('df')
    x_complete.to_csv('test_comp.csv')

if __name__=='__main__':
    #sub_pred()
    data=pd.read_csv('test_comp.csv',index_col=0)
    data=data.values
    label=pd.read_csv('label.csv',index_col=0,header=None)
    label=label.values.reshape(-1)
    print('load data')

    init_points=10
    num_iter=40

    lgbBO = BayesianOptimization(lgbccv, {'n_estimators': (500,1500),
                                            'num_leaves': (60, 120),
                                            'min_child_samples': (5,100),
                                            'reg_alpha': (0,10),
                                            'reg_lambda': (0,10),
                                            'subsample':(0.5,1.0),
                                            'colsample_bytree':(0.5,1.0)
                                            })

    lgbBO.maximize(init_points=init_points, n_iter=num_iter)
    print(lgbBO.res['max']['max_val'])
