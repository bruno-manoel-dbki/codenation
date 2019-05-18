#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 07:03:45 2019

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
from sklearn import linear_model 
from sklearn import svm
from sklearn.metrics import r2_score

classifiers = [
    #svm.SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,coef0=1),

    svm.SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1, verbose = True, max_iter = 200)
    #linear_model.SGDRegressor(),
    #linear_model.BayesianRidge(),
    #linear_model.LassoLars(),
    #linear_model.ARDRegression(),
    #linear_model.PassiveAggressiveRegressor(),
    #linear_model.TheilSenRegressor(),
    #linear_model.LinearRegression()
    ]





train = pd.read_csv("train.csv", index_col=0)
test = pd.read_csv("test.csv", index_col=0)
X = train


##LISTA DE INDICES FILTRADOS NA TABELA TEST
h = list(test.head(0))
#X = train[h]

#TROCAR STRING POR NUMEROS EM TODAS AS COLUNAS
#TRAIN
X["NU_INSCRICAO"] =pd.factorize(X.NU_INSCRICAO)[0]

#TEST
#test_ins = test.NU_INSCRICAO
#test_X = test.apply(lambda col: pd.factorize(col)[0])


#MATRZ DE CORRELAÇÕES DE TREINO
corr = X.corr()

#LINHA DE CORRELAÇÕES PARA NU_NOTAS_MT
mt_corr = corr.NU_NOTA_MT
m_corr = np.mean(abs(mt_corr.mean()))
#%%
l = list()
for i in range(len(mt_corr)):
    if abs(mt_corr[i]) > m_corr:#0.0568:
        l.append(mt_corr.index[i])

             

#%%        
X_select = train[l]
X_select = X_select.fillna(-1)


intsec = list(X_select.columns.intersection(test.columns))
X_train = X_select[intsec]


X_test = test[intsec]   
X_test = X_test.fillna(-1)


i=0
for i in range(len(intsec)):
    if (X_train.dtypes[i] != 'float64'):
        X_train[X_train.columns[i]],b = pd.factorize(X_train[X_train.columns[i]])
        X_test[X_test.columns[i]],b = pd.factorize(X_test[X_test.columns[i]])


#X_train = X_select.drop("NU_NOTA_MT", axis = 1)        



#%%
        
#a = lm.fit(X_train, X_select.NU_NOTA_MT)

#res = lm.predict(X_test)


##REGRESSoes
res = list()        
try:
    for item in classifiers:
        print(item)
        clf = item
        clf.fit(X_train, X_select.NU_NOTA_MT)
        res.append(clf.predict(X_test))
except:
    print(item)
    

#ll = list([test.index[0],res])





r = res
