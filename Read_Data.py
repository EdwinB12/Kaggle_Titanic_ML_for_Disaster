# -*- coding: utf-8 -*-
"""
Created on Mon May 11 10:34:21 2020

Kaggle Titanic: Machine Learning for a Disaster
More information is here: https://www.kaggle.com/c/titanic/data

Document Objective: Read, Simple Clean and Correlation Assessment


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

#%% -------------- Functions -------------------

#Function for reading dataset
def read_train(): 
    train_set = pd.read_csv('Original_Data/train.csv')
    return train_set

#Convenient function for dataframe QC
def Initial_Data_QC(df): 
    
    print('\n', 'shape:' , df.shape, '\n')
    print('Columns:', df.columns,'\n' )
    print(df.head(),'\n')
    print(df.tail(),'\n')
    print(df.info(),'\n')
    print(df.describe(),'\n') #Only float columns are shown here
    pass
    
#Function for replacing stirng values in a feature with a number
def feature_replace(df,feature,f_dict):
    '''
    Function to replace features that contain strings with numbers: i.e. Replace Male or Female with 1 or 2.
    Requires a user supplied dictionary containing strings needing  replacing and values to replace them. 
    
    Parameters
    ----------
    df : Pandas dataframe
        Dataframe containing feature you want replacing.
    feature : Column containing strings
        Column with string values that want replacing with numbers.
    f_dict : Dictionary
        Contains dictionary where keys are the strings that want replacing and values are the value to replace with.

    Returns
    -------
    feature_replaced : Feature Column
        Edited column that can now replace column on a dataframe.

    '''
    feature_replaced = df[feature].replace(f_dict)
    return feature_replaced

#%% ----------------- Quick clean of easy to deal with features -----------------

# Read in dataframe
train = read_train()
train_orig = train.copy()
# Replace Male with 1 and Female with 2 in 'Sex' column.
sex_dict = {'male' : 1, 'female' : 2}  
train.Sex = feature_replace(train, 'Sex', sex_dict)
#Replace C = 1, Q = 2, S = 3 in 'Embarked' column
embark_dict = {'C' : 1, 'Q' : 2, 'S':3}
train.Embarked = feature_replace(train, 'Embarked', embark_dict)
#NOTE: ONE HOT ENCODING MAY BE REQUIRED FOR THIS VARIABLE. IT IS NOT ADDRESSED HERE BUT MAY BE ON ANOTHER BRANCH

# ------------------- Assess Correlations ---------------------------------------

#Plot histogram of all non string features
train.hist(bins=20,figsize=(20,15)) # Plotting a histogram of all features

#Find features most strongly correlated to target feature
corr_matrix = train.corr()
print(corr_matrix['Survived'].sort_values(ascending=False)) 


# Write train directory into pickle format for further use in later files
with open('Pickled_Files/train_df_S1.txt', 'wb') as myFile:
    pickle.dump(train, myFile)

#with open('Pickled_Files/train_df_S1.txt', 'rb') as myFile:
    #train_df_TEST = pickle.load(myFile)

#%% Finish
    
print('\n') 
print('Script runtime:', (time.time()-start))
print('\n' '\n')
print('--------------- End --------------------')