#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 15:34:26 2019

@author: bruno_dbki
"""

     # -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as lrn
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score



train = pd.read_csv("train.csv", index_col=0)
test = pd.read_csv("test.csv", index_col=0)
X = train.copy()


#X = X.apply(lambda col: pd.factorize(col)[0])

#%%
corr = X.corr()

mt_corr = corr.NU_NOTA_MT
m_corr = mt_corr.mean()
l = list()
for i in range(len(mt_corr)):
    if abs(mt_corr[i]) > m_corr:
        l.append(mt_corr.index[i])

             

#%%        
X_select = train
X_select = X_select.fillna(-1)


intsec = list(X_select.columns.intersection(test.columns))
X_train = X_select[intsec]


X_test = test[intsec]   
X_test = X_test.fillna(-1)


i=0
for i in range(len(intsec)):
    if (X_train.dtypes[i] == 'object'):
        X_train[X_train.columns[i]],b = pd.factorize(X_train[X_train.columns[i]])
        X_test[X_test.columns[i]],b = pd.factorize(X_test[X_test.columns[i]])


#X_train = X_select.drop("NU_NOTA_MT", axis = 1)        



#%%

##REGRESSAO LINEAR
#
        
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
# Don't cheat - fit only on training data
scaler.fit(X_train)  
X_train = scaler.transform(X_train)  
# apply same transformation to test data
X_test = scaler.transform(X_test)          

#%%

#lm = MLPRegressor(alpha=0.1)#93.55
#lm = MLPRegressor(alpha=0.01)#93.6
#lm = MLPRegressor(alpha=0.05)

lm = MLPRegressor(
    hidden_layer_sizes= (800,), 
    #hidden_layer_sizes= (93,79,22),
    activation="", 
    #solver="adam", 
    alpha=0.01, 
    batch_size= 250, 
    max_iter = 200,
    learning_rate="constant", 
    #tol = 1e-6,
    #learning_rate_init=0.05, 
    
    #power_t=0.9, 
    #beta_1 = 0.009,
    #beta_2 = 0.99,
    #momentum=0.9, 
    #epsilon=1e-8,
    verbose=1, 
    early_stopping=True
    ) 

a = lm.fit(X_train, X_select.NU_NOTA_MT)
#
res = lm.predict(X_test)

#ll = list([test.index[0],res])

r = res

r[r<320]=0
#%%
out = test.copy()[[]]

out["NU_NOTA_MT"] = r
    
out.to_csv("answer.csv")
