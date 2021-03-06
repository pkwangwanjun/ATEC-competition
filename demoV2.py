# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
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
sys.path.append('/Users/wanjun/Desktop/LightGBM/python-package')
import lightgbm as lgb

import gc

import mlxtend
from sklearn.metrics import confusion_matrix,roc_curve

from bayes_opt import BayesianOptimization
from sklearn.model_selection import StratifiedKFold

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

import warnings
warnings.filterwarnings("ignore")


#994731
#print(len(data))
#print(len(np.unique(data.index)))
 #0    977884
 #1     12122
#-1      4725
#pd.value_counts(data['label'])


#记分函数
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
    data=pd.read_csv('data/atec_anti_fraud_train.csv',index_col=0)

    data.loc[data.label==-1,'label']=1
    #data=data[data['label']!=-1]
    data.fillna(-1,inplace=True)
    y=data['label']
    print(pd.value_counts(y))
    x=data[data.columns[data.columns!='label']]
    x.drop('date',axis=1,inplace=True)
    #for i in range(1,20):
    #    if i==5:
    #        continue
    #    x['f{}'.format(i)]=x['f{}'.format(i)].astype('category')
    gc.collect()

    lgbc=lgb.LGBMClassifier(n_estimators=500,max_depth=-1,num_leaves=100,learning_rate=0.01,subsample=0.8,sub_feature=0.8,random_state=0,n_jobs=-1,objective='binary')
    lgbc.fit(x,y)
    test=pd.read_csv('data/atec_anti_fraud_test_b.csv',index_col=0)
    test.fillna(-1,inplace=True)
    test.drop('date',axis=1,inplace=True)
    #for i in range(1,20):
    #    if i==5:
    #        continue
    #    test['f{}'.format(i)]=test['f{}'.format(i)].astype('category')
    pred_prob=lgbc.predict_proba(test)
    temp=pd.DataFrame(pred_prob[:,1],index=test.index)
    temp.reset_index(inplace=True)
    temp.columns=['id','score']
    temp.to_csv('test_wtf_b.csv',index=None)


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

'''
 colsample_bytree：0.9547
 min_child_samples 5.3656
 n_estimators：1450.1096
num_leaves 119.3153 |
reg_alpha 0.1930 |
reg_lambda： 8.6361 |
subsample 0.8495 |
'''

def obsearch():
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


def statis_data():
    #每个id唯一
    data=pd.read_csv('data/atec_anti_fraud_train.csv',index_col=0)
    data.sort_values('date',inplace=True)


    data.loc[data.label==-1,'label']=1
    #62天
    print(np.unique(data.date))

    num_all=data.groupby('date').apply(lambda x:len(x))
    idx=np.arange(len(num_all))
    p1=plt.bar(idx,num_all.values.reshape(-1))
    plt.xlabel('date')
    plt.ylabel('number')
    plt.savefig('trade.jpg')
    plt.clf()

    num_bad=data[data.label==1].groupby('date').apply(lambda x:len(x))
    idx=np.arange(len(num_bad))
    p1=plt.bar(idx,num_bad.values.reshape(-1))
    plt.xlabel('date')
    plt.ylabel('number')
    plt.savefig('bad.jpg')
    plt.clf()

    test=pd.read_csv('data/atec_anti_fraud_test_b.csv',index_col=0)

    #f36-f47
    #16627个bad f36-f47为nan
    (data.f36==-1)&(data.f37==-1)&(data.f38==-1)&(data.f39==-1)&(data.f40==-1)&(data.f41==-1)&(data.f42==-1)&(data.f43==-1)&(data.f44==-1)&(data.f45==-1)&(data.f46==-1)&(data.f47==-1)


def select_sample():
    test=pd.read_csv('data/atec_anti_fraud_test_b.csv',index_col=0)
    train=pd.read_csv('data/atec_anti_fraud_train.csv',index_col=0)

    train.fillna(-1,inplace=True)
    #f1-f19 int 型
    max_lst=[]
    min_lst=[]
    for i in range(20,298):
        max_lst.append(np.max(test['f{}'.format(i)]))
        min_lst.append(np.min(test['f{}'.format(i)]))

    for index,(max_num,min_num) in enumerate(zip(max_lst,min_lst)):
        train=train[((train['f{}'.format(index+20)]<=max_num) & (train['f{}'.format(index+20)]>=min_num)) | (train['f{}'.format(index+20)]==-1)]

    return train


def vali(x,y):
    lgbc=lgb.LGBMClassifier(n_estimators=500,max_depth=-1,num_leaves=100,learning_rate=0.01,subsample=0.8,sub_feature=0.8,random_state=0,n_jobs=-1,objective='binary',is_unbalance=True)
    lgbc.fit(x,y)
    precision_score(y,lgbc.predict(x))
    roc_auc_score(y,lgbc.predict_proba(x)[:,1])
    recall_score(y,lgbc.predict(x))



train_0=pd.read_csv('train_0.csv',index_col=0)
train_1=pd.read_csv('train_1.csv',index_col=0)

#train=train_0.iloc[np.random.choice(range(len(train_0)),size=13*len(train_1),replace=False)]
data=train_0.append(train_1)
X=data[data.columns.difference(['label'])]
X.drop('date',axis=1,inplace=True)
Y=data.label
lgbc=lgb.LGBMClassifier(n_estimators=500,max_depth=-1,class_weight={0:1,1:1.5},num_leaves=100,learning_rate=0.01,subsample=0.8,sub_feature=0.8,random_state=0,n_jobs=-1,objective='binary')
lgbc.fit(X,Y)

test=pd.read_csv('data/atec_anti_fraud_test_b.csv',index_col=0)
test.fillna(-1,inplace=True)
test.drop('date',axis=1,inplace=True)

pred_prob=lgbc.predict_proba(test)
temp=pd.DataFrame(pred_prob[:,1],index=test.index)
temp.reset_index(inplace=True)
temp.columns=['id','score']
temp.to_csv('test_wtf_b.csv',index=None)

if __name__=='__main__':
    sub_pred()
    #pass
