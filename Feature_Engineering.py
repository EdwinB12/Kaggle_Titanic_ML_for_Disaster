# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:14:26 2020

Kaggle Titanic: Machine Learning for a Disaster
More information is here: https://www.kaggle.com/c/titanic/data

Document Objective: Feature Engineering

@author: edwin
"""

# ------------------ Import Packages ----------------------

import numpy as np
import seaborn as sns
import pandas as pd
import time
import MLFunLib as mlib # Custom made library
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


print('\n')
print('--------------- Start --------------------')
print('\n' '\n')
start=time.time()

# ----------------- Reading data in -----------------------------

# Reading training data
path = 'Original_Data/train.csv'
train_original =  pd.read_csv(path)
train_df = train_original.copy()


# -----------------------------------------------------------------------------
# ----------------- Feature Engineering ---------------------------------------

# Drop Passenger ID and Cabin 
train_df = train_df.drop(['PassengerId','Ticket','Cabin'],axis=1)

# ---------------- Create 'Title' feature and remove 'Name' ------------------

# Extract Title from Name
train_df['Title'] = train_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# List all Titles by Sex
print(pd.crosstab(train_df.Title, train_df.Sex))

# Group any title that isn't Mr, MRs, Miss or Master in group 'other'. 1st correct some typos etc. 
train_df['Title'] = train_df['Title'].replace('Mlle', 'Miss')
train_df['Title'] = train_df['Title'].replace('Ms', 'Miss')
train_df['Title'] = train_df['Title'].replace('Mme', 'Mrs')
train_df['Title'] = train_df['Title'].replace(['Don', 'Rev', 'Dr', 'Mme', 'Ms',
       'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'Countess',
       'Jonkheer'],'Other')

# Remove Name Column
train_df = train_df.drop('Name',axis=1)

# ----------------- Sorting Age into Brackets ----------------

# Bin age into discrete values and look at mean survival
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

# Print mean survival value for each AgeBand
print(
      train_df[['AgeBand','Survived']].groupby('AgeBand').mean().sort_values(by='AgeBand',ascending=True))

# Define New Column AgeInt and assign values based on the Ageband analysis above. Remove Ageband.
train_df['AgeInt'] = train_df['Age']
   
train_df.loc[ train_df['Age'] <= 16, 'AgeInt'] = 0
train_df.loc[(train_df['Age'] > 16) & (train_df['Age'] <= 32), 'AgeInt'] = 1
train_df.loc[(train_df['Age'] > 32) & (train_df['Age'] <= 48), 'AgeInt'] = 2
train_df.loc[(train_df['Age'] > 48) & (train_df['Age'] <= 64), 'AgeInt'] = 3
train_df.loc[train_df['Age'] > 64, 'AgeInt'] = 4
train_df = train_df.drop('AgeBand',axis=1)

# ---------------- IsAlone Feature - 1 if alone, 0 if not -------------------------

# Assign all rows 0 as IsAlone value. 
train_df['IsAlone'] = 0

# Make IsAlone = 1 if SibSp and Parch both = 0
train_df.loc[(train_df['SibSp'] == 0) & (train_df['Parch'] <= 0), 'IsAlone'] = 1
    
# ----------------------------- Fare Bands ----------------------------

#train_df['FareBand'] = pd.cut(train_df['Fare'],50)
#train_df[['FareBand','Survived']].groupby['FareBand'].mean()
train_df.loc[ train_df['Fare'] <= 10, 'FareInt'] = 0
train_df.loc[(train_df['Fare'] > 10) & (train_df['Fare'] <= 30), 'FareInt'] = 1
train_df.loc[(train_df['Fare'] > 30) , 'FareInt'] = 2
print(train_df[['FareInt','Survived']].groupby('FareInt').count())
print(train_df[['FareInt','Survived']].groupby('FareInt').mean())

# This is now ready to go into the Pipeline. 

#NOTE! : This is now written to a function in 'mlib.Feature_Engineering'. 
#This script has a little extra analysis added. 

# ------------------------ Finish --------------------------------------------
    
print('\n') 
print('Script runtime:', (time.time()-start)/60)
print('\n' '\n')
print('----------------- End --------------------')