# -*- coding: utf-8 -*-
"""
Created on Wed May 13 18:27:59 2020

Kaggle Titanic: Machine Learning for a Disaster
More information is here: https://www.kaggle.com/c/titanic/data

Document Objective: Submission_1

Workflow: 
    1. Choose six features from training data:'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked'
    2. Made two seperate pipes: 
        - Numerical Pipe: Replace missing values with mean values (1 value only in the test dataset); Standardize feature scaling
        - Categorical Pipe: Replace missing values with most common (2 instances for 'Embarked' class); One hot encoding 
    3. Combine pipes into a full pipeline and push training and test data through. 
    4. Train SVM classifier with rbf kernel. This had been preferred from previous testing on a similarly conditioned dataset
    5. Quick 10 fold cross validation evaluation, optional choice to lot learning curve.
    6. Write out to CSV file and submit to Kaggle. - Submission 1. Score - 75%
    
Things to try for future submissions: 
    Grid Search for SV parameters 
    Try new algorithms - Random Forest Perhaps 
    Proper validation to see what instances it is struggling with. 
    Finally, revisit pre-processing of data - but do this last. See max achieviable score with these pre-processing pipelines. 
    
@author: edwin
"""


# ------------------ Import Packages ----------------------

# General Packages
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pydot
import time

# Sklearn
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import cross_val_score, learning_curve, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier,GradientBoostingClassifier

# Custom 
import MLFunLib as mlib # Custom made library

print('\n')
print('--------------- Start --------------------')
print('\n' '\n')
start=time.time()



# --------------- Data Read, Feature Engineering and data prep for training ---------------------

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
    ('ohe' , OneHotEncoder())
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



# -------------------------- SVM classifier rbf ------------------------------

# Initialise SVM parameters
svm_clf_rbf_C100 = SVC(kernel='rbf',C=100,random_state=42)

# Fit SVM to data
svm_clf_rbf_C100.fit(train_features_prep,target_features)

#Output best Cross Validation score
print('SVM CVS score:' , np.mean(cross_val_score(svm_clf_rbf_C100, train_features_prep,target_features,cv=5,
                scoring = 'accuracy')))
print('SVM Done!','\n')



# ----------------- Plotting Learning Curve for best result -------------------

# Plotting Learning Curve
fig,ax = plt.subplots(figsize=(12,8))
ax.set_title('Submission 1 - SVM Learning Curve')
ax.set_ylim(0.6,1.02)
ax.set_xlim(0,720)
mlib.plot_learning_curve(svm_clf_rbf_C100, train_features_prep, target_features, 'accuracy', 5, ax)



# ---------------- Predicting on test data and save submission ---------------

# # Predict on prepared test dataset and write out to csv
# predictions=svm_clf_rbf_C100.predict(test_features_prep)
# mlib.Pred_to_Kaggle_Format(predictions,'Submissions/Submission_1.csv')



# ------------------------ Finish --------------------------------------------
    
print('\n') 
print('Script runtime:', (time.time()-start)/60)
print('\n' '\n')
print('----------------- End --------------------')