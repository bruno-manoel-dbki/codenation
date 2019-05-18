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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn import svm



train = pd.read_csv("train.csv", index_col=0)
test = pd.read_csv("test.csv", index_col=0)

X = train.copy()





#MATRZ DE CORRELAÇÕES DE TREINO
corr = X.corr()

#LINHA DE CORRELAÇÕES PARA NU_NOTAS_MT

mt_corr = corr.NU_NOTA_MT
m_corr = np.mean(abs(mt_corr.mean()))
#%%
l = list()
for i in range(len(mt_corr)):
    if abs(mt_corr[i]) > 0.0168:
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

##REGRESSAO LINEAR
#
        
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()  
# Don't cheat - fit only on training data
scaler.fit(X_train)  
X_train = scaler.transform(X_train)  
# apply same transformation to test data
X_test = scaler.transform(X_test)          

#%%

##REGRESSAO LINEAR
#
#lm = LinearRegression()
#gamma = 0.15 -> 93,37%
#lm = svm.SVR(kernel='rbf', C=50, gamma='scale', epsilon=.01, verbose = True, max_iter = 5000) #93.45%

lm = svm.SVR(kernel='rbf', C=100, gamma='scale', epsilon=.001, verbose = True, max_iter = 1000) 
a = lm.fit(X_train, X_select.NU_NOTA_MT)
#
res = lm.predict(X_test)

#ll = list([test.index[0],res])

r = res

r[r<320]=0

out = test.copy()[[]]

out["NU_NOTA_MT"] = r

out.to_csv("answer.csv")
