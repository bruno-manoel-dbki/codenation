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



train = pd.read_csv("train.csv", index_col=0)
test = pd.read_csv("test.csv", index_col=0)
# FIRST THING TO DO IS CONCATENATE GRADES


#train["NU_NOTA_MT"] = train["NU_NOTA_MT"]*3

#train["NU_NOTA_CN"] = train["NU_NOTA_CN"]*2

#train["NU_NOTA_LC"] = train["NU_NOTA_LC"]*1.5

#train["NU_NOTA_CH"] = train["NU_NOTA_CH"]*1

#train["NU_NOTA_REDACAO"] = train["NU_NOTA_REDACAO"]*3


#NU_NOTA_MT IS THE ANSWER

#sns.distplot(test["NU_NOTA_CN"].dropna())
#sns.distplot(test["NU_NOTA_LC"].dropna())
#sns.distplot(test["NU_NOTA_CH"].dropna())
#sns.distplot(test["NU_NOTA_REDACAO"].dropna())
#plt.figure()
#sns.distplot(train["NU_NOTA_CN"].dropna())
#sns.distplot(train["NU_NOTA_LC"].dropna())
#a = sns.distplot(train["NU_NOTA_CH"].dropna())
#sns.distplot(train["NU_NOTA_REDACAO"].dropna())
#
#train = train.fillna(0)
#test = test.fillna(0)
#X = train.drop("NU_NOTA_MT", axis = 1)
X = train


##LISTA DE INDICES FILTRADOS NA TABELA TEST
h = list(test.head(0))
#X = train[h]

#TROCAR STRING POR NUMEROS EM TODAS AS COLUNAS
#TRAIN
X = X.apply(lambda col: pd.factorize(col)[0])

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
    if abs(mt_corr[i]) > 0.0568:
        l.append(mt_corr.index[i])

             

#%%        
X_select = train[l]
X_select = X_select.fillna(0)


intsec = list(X_select.columns.intersection(test.columns))
X_train = X_select[intsec]


X_test = test[intsec]   
X_test = X_test.fillna(0)


i=0
for i in range(len(intsec)):
    if (X_train.dtypes[i] != 'float64'):
        X_train[X_train.columns[i]],b = pd.factorize(X_train[X_train.columns[i]])
        X_test[X_test.columns[i]],b = pd.factorize(X_test[X_test.columns[i]])


#X_train = X_select.drop("NU_NOTA_MT", axis = 1)        



#%%

##REGRESSAO LINEAR
#
lm = LinearRegression()
#
a = lm.fit(X_train, X_select.NU_NOTA_MT)
#
res = lm.predict(X_test)

ll = list([test.index[0],res])

r = res

r[r<50]=0

out = test.copy()[[]]

out["NU_NOTA_MT"] = r

out.to_csv("answer.csv")
