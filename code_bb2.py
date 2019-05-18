     # -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score



train = pd.read_csv("train.csv", index_col=0)
test = pd.read_csv("test.csv", index_col=0)
X_train = train.copy()
X_test = test.copy()



#%%
i=0
for i in range(len(X_train.columns)):
    if (X_train.dtypes[i] == 'object'):
        X_train[X_train.columns[i]],b = pd.factorize(X_train[X_train.columns[i]])
i=0
for i in range(len(X_test.columns)):
    if (X_test.dtypes[i] == 'object'):        
        X_test[X_test.columns[i]],b = pd.factorize(X_test[X_test.columns[i]])


X_train = X_train.fillna(0)

X_test = X_test.fillna(0)


intsec = list(X_train.columns.intersection(X_test.columns))

X_select = X_train[intsec]

X_test = X_test[intsec]  


 
#%%
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
# Don't cheat - fit only on training data
scaler.fit(X_select)  
X_select = scaler.transform(X_select)  
# apply same transformation to test data
X_test = scaler.transform(X_test)          
             

#%%        
        



#%%

##REGRESSAO LINEAR
#
        

#%%

#lm = MLPRegressor(alpha=0.1)#93.55
#lm = MLPRegressor(alpha=0.01)#93.6
#lm = MLPRegressor(alpha=0.05)

lm = MLPRegressor(
    hidden_layer_sizes= (50,20,10,), 
    #hidden_layer_sizes= (26,15,10),
    #activation="relu", 
    #solver="adam", 
    alpha=0.01, 
    batch_size= 250, 
    max_iter = 200,
    learning_rate="constant", 
    tol = 1e-6,
    #learning_rate_init=0.01, 
    #power_t=0.5, 
    #beta_1 = 0.009,
    #beta_2 = 0.99,
    #momentum=0.9, 
    #epsilon=1e-8,
    verbose=1, 
    early_stopping=True
    ) 

a = lm.fit(X_select, X_train.NU_NOTA_MT)
#
res = lm.predict(X_test)

#ll = list([test.index[0],res])

r = res

r[r<320]=0
#%%
out = test.copy()[[]]

out["NU_NOTA_MT"] = r
    
out.to_csv("answer.csv")
