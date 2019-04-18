     # -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
# FIRST THING TO DO IS CONCATENATE GRADES


#train["NU_NOTA_MT"] = train["NU_NOTA_MT"]*3

#train["NU_NOTA_CN"] = train["NU_NOTA_CN"]*2

#train["NU_NOTA_LC"] = train["NU_NOTA_LC"]*1.5

#train["NU_NOTA_CH"] = train["NU_NOTA_CH"]*1

#train["NU_NOTA_REDACAO"] = train["NU_NOTA_REDACAO"]*3


#NU_NOTA_MT IS THE ANSWER

sns.distplot(test["NU_NOTA_CN"].dropna())
sns.distplot(test["NU_NOTA_LC"].dropna())
sns.distplot(test["NU_NOTA_CH"].dropna())
sns.distplot(test["NU_NOTA_REDACAO"].dropna())
plt.figure()
sns.distplot(train["NU_NOTA_CN"].dropna())
sns.distplot(train["NU_NOTA_LC"].dropna())
sns.distplot(train["NU_NOTA_CH"].dropna())
sns.distplot(train["NU_NOTA_REDACAO"].dropna())

