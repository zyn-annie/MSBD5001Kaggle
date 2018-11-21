'''
Created 2018-10-27
XGBoost Algorithm on predicting time
Author: ZHANG Yuning
'''
#!/usr/bin/python
# coding:utf8

import numpy as np
import pandas as pd
import sklearn.feature_selection
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import Imputer
from sklearn import metrics
from sklearn.decomposition import PCA 

import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
#from __future__ import print_function
#����������ȷ��
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.preprocessing import scale
from xgboost import plot_importance


def loadcsv():
    df = pd.read_csv('train.csv')
    dftrain = pd.read_csv('train.csv',usecols=[1,3,4,6,7,8,9,10,11,12,13])
    #print(dftrain.head())
    dflabel= pd.read_csv('train.csv',usecols=[14])
    #print(dflabel.head())
    df2 = pd.read_csv('test.csv')
    dftest=pd.read_csv('test.csv',usecols=[1,3,4,6,7,8,9,10,11,12,13])
    #print(dftest.head())
    return dftrain,dftest,dflabel,df2

def preprocess(dftrain,dftest):
    #�����±�������labelEncoder����
    dftrain['penalty']=dftrain['penalty'].replace('none',0)
    dftrain['penalty']=dftrain['penalty'].replace('l2',1)
    dftrain['penalty']=dftrain['penalty'].replace('elasticnet',2)
    dftrain['penalty']=dftrain['penalty'].replace('l1',3)
    dftest['penalty']=dftest['penalty'].replace('none',0)
    dftest['penalty']=dftest['penalty'].replace('l2',1)
    dftest['penalty']=dftest['penalty'].replace('elasticnet',2)
    dftest['penalty']=dftest['penalty'].replace('l1',3)
    dftrain['new1']=dftrain['n_samples']*dftrain['n_features']/dftrain['n_jobs']
    dftest['new1']=dftest['n_samples']*dftest['n_features']/dftest['n_jobs']
    dftrain['new2']=dftrain['n_samples']*dftrain['n_features']
    dftest['new2']=dftest['n_samples']*dftest['n_features']
    dftrain=pd.DataFrame(dftrain)
    #print(dftrain.head())
    dftest=pd.DataFrame(dftest)
    #rint(dftest.head())
    return dftrain,dftest

def featureSet(df): 
    df['alpha'] = -(np.log10(df['alpha']))
    df['n_jobs'] = df['n_jobs'].replace(-1,max(df['n_jobs']))
    return df


def trainXGBoost(dftrain,dftest,dflabel,df2):
    
    x_train,x_test,y_train,y_test=train_test_split(dftrain.values, dflabel.values, test_size=0.15,random_state=66)
    df=pd.DataFrame(x_train)
    #print(df.head())
    #print(dftest.head())
    XGB = XGBRegressor(silent=1 ,#���ó�1��û��������Ϣ��������������Ϊ0.�Ƿ�����������ʱ��ӡ��Ϣ��
                       #booster='gblinear',#��������ģ�ͽ����������㣬Ĭ����gbtree
                       learning_rate=0.2, # ��ͬѧϰ��
                        min_child_weight=1, 
                        # �������Ĭ���� 1����ÿ��Ҷ������ h �ĺ������Ƕ��٣�����������������ʱ�� 0-1 �������
                        #������ h �� 0.01 ������min_child_weight Ϊ 1 ��ζ��Ҷ�ӽڵ���������Ҫ���� 100 ��������
                        #��������ǳ�Ӱ����������Ҷ�ӽڵ��ж��׵��ĺ͵���Сֵ���ò���ֵԽС��Խ���� overfitting��
                        max_depth=4, # ����������ȣ�Խ��Խ���׹����
                        gamma=0.2,  # ����Ҷ�ӽڵ�������һ�������������С��ʧ����,Խ��Խ���أ�һ��0.1��0.2�����ӡ�
                        subsample=1, # �������ѵ������ ѵ��ʵ�����Ӳ�����
                        max_delta_step=1,#���������������������ÿ������Ȩ�ع��ơ�
                        colsample_bytree=1, # ������ʱ���е��в��� 
                        reg_lambda=6,  # ����ģ�͸��Ӷȵ�Ȩ��ֵ��L2���������������Խ��ģ��Խ�����׹���ϡ�
                        reg_alpha=0, # L1 ���������
                        scale_pos_weight=1, #���ȡֵ����0�Ļ��������������ƽ�������������ڿ���������ƽ������Ȩ��
                        n_estimators=360, #���ĸ���
                        seed=60 #�������
                      )
    XGB.fit(x_train,y_train,eval_metric='rmse',verbose=False)
    y_true, y_pred = y_test, XGB.predict(x_test)
    ans=XGB.predict(dftest.values)
    
    for t in range(len(dftest)):
        if ans[t]<=0:
            ans[t]=min(dflabel.values)
         
    submission_df = pd.DataFrame(data = {'Id':df2.id,'time':ans})
    submission_df.to_csv('submission.csv',index=None)

    #print('Mean Absolute Error:', metrics.mean_absolute_error(y_true, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_true, y_pred))  
    #print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_true, y_pred)))
    
    plot_importance(XGB)
    plt.show()

 

    
if __name__ == "__main__":
    dftrain,dftest,dflabel,df2 = loadcsv()
    dftrain=featureSet(dftrain)
    dftest=featureSet(dftest)
    dftrain,dftest=preprocess(dftrain,dftest)
    trainXGBoost(dftrain,dftest,dflabel,df2)