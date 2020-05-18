# -*- coding: utf-8 -*-
"""
Created on Sun May 17 14:14:40 2020

Data Vis Testing

Script for Experimenting with data visualisation and PCA.  

@author: edwin
"""

# ------------------ Import Packages ----------------------


import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier, LogisticRegression
import time
from sklearn.model_selection import cross_val_score, learning_curve, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import MLFunLib as mlib # Custom made library
from sklearn.tree import DecisionTreeClassifier, export_graphviz 
import pydot
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

print('\n')
print('--------------- Start --------------------')
print('\n' '\n')
start=time.time()



# -------------------- Preparing data -------------------

# Prep data for Pipes - Training Data
path = 'Original_Data/train.csv'
train_original = mlib.csv_to_df(path)
feature_list = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']
target_list = ['Survived']

# Define Numeric Pipeline - Standard Scaling
num_pipe = Pipeline([
    ('imputer_mean', SimpleImputer(strategy='mean')),
    ('std_scalar', StandardScaler())
])

# Define Categorical Pipeline - One Hot Enconding
cat_pipe = Pipeline([
    ('imputer' , SimpleImputer(strategy = 'most_frequent')),
    #('ohe' , OneHotEncoder()),
    ('oe', OrdinalEncoder())
    
])

#Combinin Pipes into full pipeline 
full_pipeline,train_features,target_features, post_transform_train_features = mlib.Full_PipeLine(
    path,feature_list,target_list,num_pipe, cat_pipe)

# Transform data using final combined pipeline
train_features_prep = full_pipeline.fit_transform(train_features)


# ----------------------------- PCA Analysis ---------------------------------

# Printing variance ratio's
pca = PCA()
pca.fit(train_features_prep)
print(pca.explained_variance_ratio_) # Printing variance ratios

# Plotting Scree Plot
mlib.Scree_Plot(train_features_prep)

# Scatter Matrix of PCs
mlib.PC_CrossPlotting_Color(train_features_prep, target_features)

# ------------------------ Finish --------------------------------------------
    
print('\n') 
print('Script runtime:', (time.time()-start)/60)
print('\n' '\n')
print('----------------- End --------------------')


# -------------------------- Bits and Bobs --------------------------------

# sns.catplot(
#     x= 'Sex', hue = 'Embarked', col = 'Survived', data = train_features_prep_df, kind='count')
# sns.catplot(x="Sex", y="Survived", col="Pclass",
#                 data=train_features_prep_df, saturation=.5,
#                 kind="bar", ci=None, aspect=.6)