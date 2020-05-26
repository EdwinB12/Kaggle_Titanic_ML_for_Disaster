# -*- coding: utf-8 -*-
"""
Created on Sun May 17 14:14:40 2020

Data Vis Testing

Principal Components Analysis (PCA) on Prep1 and Prep2.  

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


# ------------------------------ Prep 1 ---------------------------------------

# Read Train and Test Datasets and save off original copies
path = 'Original_Data/train.csv'
train_original = mlib.csv_to_df(path)
train_df = train_original.copy()
path = 'Original_Data/test.csv'
test_original = mlib.csv_to_df(path)
test_df = test_original.copy()

# # Create list of features desired for training
feature_list = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']
target_list = ['Survived']

# Define Numeric Pipeline
num_pipe = Pipeline([
    ('imputer_mean', SimpleImputer(strategy='mean')),
    ('std_scalar', StandardScaler())
])

# Define Categorical Pipeline
cat_pipe = Pipeline([
    ('imputer' , SimpleImputer(strategy = 'most_frequent')),
    # ('ohe' , OneHotEncoder()),
    ('oe', OrdinalEncoder())
])

#Combining Pipes into full pipeline - Train Data
full_pipeline,train_features,target_features,post_trans_train_feature = mlib.Full_PipeLine(
    train_df,feature_list,target_list,num_pipe,cat_pipe)

# Combining Pipes into full pipeline - Test Data
full_pipeline_test,test_features,empty, post_transform_test_features = mlib.Full_PipeLine(
    test_df,feature_list,[],num_pipe,cat_pipe)

# Transform data using final combined pipeline - Train
train_features_prep = full_pipeline.fit_transform(train_features)

# Transform data using final combined pipeline - Test
test_features_prep = full_pipeline.fit_transform(test_features)



# ----------------------------- PCA Analysis ---------------------------------

# Printing variance ratio's
pca = PCA()
pca.fit(train_features_prep)
print('Prep1:' , pca.explained_variance_ratio_) # Printing variance ratios

# Plotting Scree Plot
fig,ax = plt.subplots(figsize= (12,8))
ax.set_title('Prep 1: Scree Plot')
mlib.Scree_Plot(train_features_prep,ax)

# Scatter Matrix of PCs
g = mlib.PC_CrossPlotting_Color(train_features_prep, target_features,ax)
g.fig.suptitle('Prep 1: Principal Component Cross Plot ',y=1.04,fontsize='xx-large')




#----------------------------- Prep 2 ----------------------------------------

# Read Train and Test Datasets and save off original copies
train_path = 'Original_Data/train.csv'
train_original = pd.read_csv(train_path)
train_df = train_original.copy()
test_path = 'Original_Data/test.csv'
test_original = pd.read_csv(test_path)
test_df = test_original.copy()

# Feature Engineering 
train_df = mlib.Feature_Engineering(train_df)
test_df = mlib.Feature_Engineering(test_df)

# Create list of features desired for training
feature_list = ['Pclass', 'Sex', 'AgeInt', 'IsAlone', 'FareInt', 'Embarked','Title']
target_list = ['Survived']

# Define Numeric Pipeline
num_pipe = Pipeline([
    ('imputer_mean', SimpleImputer(strategy='most_frequent')),
    #('std_scalar', StandardScaler())
])

# Define Categorical Pipeline 
cat_pipe = Pipeline([
    ('imputer' , SimpleImputer(strategy = 'most_frequent')),
    #('ohe' , OneHotEncoder()),
    ('oe', OrdinalEncoder())
    
])

# Combining Pipes into full pipeline - Training Data
full_pipeline_train,train_features,target_features, post_transform_train_features = mlib.Full_PipeLine(
    train_df,feature_list,target_list,num_pipe, cat_pipe)

# Combining Pipes into full pipeline - Test Data
full_pipeline_test,test_features,empty, post_transform_test_features = mlib.Full_PipeLine(
    test_df,feature_list,[],num_pipe,cat_pipe)

# Transform data using final combined pipeline
train_features_prep = full_pipeline_train.fit_transform(train_features)
test_features_prep = full_pipeline_train.fit_transform(test_features)





# ---------------------------------------- PCA Prep 2 -------------------------

# Printing variance ratio's
pca = PCA()
pca.fit(train_features_prep)
print('Prep 2:' ,pca.explained_variance_ratio_) # Printing variance ratios

# Plotting Scree Plot
fig,ax = plt.subplots(figsize= (12,8))
ax.set_title('Prep 2: Scree Plot')
mlib.Scree_Plot(train_features_prep,ax)

# Scatter Matrix of PCs
g = mlib.PC_CrossPlotting_Color(train_features_prep, target_features,ax)
g.fig.suptitle('Prep 2: Principal Component Cross Plot ',y=1.04,fontsize='xx-large')


# # Form prepped data into df if required. 
# corr_df = pd.DataFrame(data = train_features_prep, columns = post_transform_test_features)
# corr_df['Survived'] = target_features


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