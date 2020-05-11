# -*- coding: utf-8 -*-
"""
Created on Mon May 11 13:42:52 2020

Kaggle Titanic: Machine Learning for a Disaster
More information is here: https://www.kaggle.com/c/titanic/data

Document Objective: Remove NaNs, feature engineering and prepare input into ML.

@author: edwin

"""

#%% ------------- Packages and start of script ----------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import time
from pandas.plotting import scatter_matrix
import pickle
print('\n')
print('--------------- Start --------------------')
print('\n' '\n')
start=time.time()
#%%

with open('Pickled_Files/train_df_S1.txt', 'rb') as myFile:
    train_df = pickle.load(myFile)

# Output info on dataframe features
print(train_df.info())
train_df_S1 = train_df.copy() #Saving off unedited version for future comparison if needed

# ----------------- Feature Selection ---------------------- 

# Name, Ticket and Cabin are object types and are therefore discarded at this stage (likely to be unhelpful). 
# Future branches of this project may further process these features if they are thought to be helpful. 
train_df = train_df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# ----------------- Dealing with NaNs ----------------------
# 'Age' feature is missing 177 instances. Is there a correlation to Survival if you remove NaNs?  
train_df_tmp = train_df.dropna(subset = ['Age']) # Remove instances with NaNs and write to temp dataframe
corr_matrix = train_df_tmp.corr()
print('\n', 'R value between Age and survival is:' , corr_matrix['Survived'].Age) # Little evidence of correlation
train_df_tmp.plot('Age','Survived',kind = 'scatter')

# Age is removed from the df as there is little evidence of correlation
train_df = train_df.drop(['Age'], axis=1)

#Embarked has two NaN values. As it is only 2 instances, they are removed from the dataframe. 
train_df = train_df.dropna(subset = ['Embarked']) # Remove instances with NaNs in column 'Embarked'
train_df.Embarked = train_df.Embarked.astype('int64') #Convert Embarked to Int64

# ---------------- Feature Scaling -----------------------






















#%%
print('\n') 
print('Script runtime:', (time.time()-start))
print('\n' '\n')
print('--------------- End --------------------')