# -*- coding: utf-8 -*-
"""
Created on Mon May 11 13:42:52 2020

Kaggle Titanic: Machine Learning for a Disaster
More information is here: https://www.kaggle.com/c/titanic/data

Document Objective: Remove NaNs, feature engineering and prepare to input into ML.

@author: edwin

"""

#%% ------------- Packages and start of script ----------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import time
from pandas.plotting import scatter_matrix
print('\n')
print('--------------- Start --------------------')
print('\n' '\n')
start=time.time()
#%%

#train.astype({'Embarked':'int64'})
#g=train.Embarked.isnull()
#print(train[g])























#%%
print('\n') 
print('Script runtime:', (time.time()-start))
print('\n' '\n')
print('--------------- End --------------------')