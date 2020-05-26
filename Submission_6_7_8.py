# -*- coding: utf-8 -*-
"""
Created on Mon May 18 09:21:35 2020


Kaggle Titanic: Machine Learning for a Disaster
More information is here: https://www.kaggle.com/c/titanic/data

Document Objective: Submission_6,7,8

Workflow: 
    1. Generate new features. 
    2. Choose 7 features from training data:'Pclass', 'Sex', 'AgeInt', 'IsAlone', 'FareInt', 'Embarked','Title'
    2. Made two seperate pipes: 
        - Numerical Pipe: Replace missing values with most common. 
        - Categorical Pipe: Ordinal Encoding
    3. Combine pipes into a full pipeline and push training and test data through. 
    4. Initiliase models: 
        Submission 6 - SGD Classifier
        Submission 7 - Decision Tree
        Submission 8 - Random Forest
    5. GridSearch for all 3 algorithms
    6. Print best parameters and score
    8. Plot learning curve of each algorithm
    9. Write out to CSV file and submit to Kaggle.
        - Submission 6. Score - 78.4%
        - Submission 7. Score - 77.5%
        - Submission 8. Score - 78.9%
 
   For Final Parameters, check Kaggle submissions or run script. 
     

Things to try for future submissions: 
    Proper validation to see what instances it is struggling with. 
    Boosting! 
    Try reducing lesser important parameters
    
    
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



# -------------------------------------------------------------------------------------------------
# -------------------------------- Training -------------------------------------------------

# The following lines will initialise the same methods used in Submission 1,2 and 5. 

# Submission 1
svm_clf = SVC(random_state=42)

# Submission 2
tree_clf = DecisionTreeClassifier(random_state=42) 

# Submission 5
rfc = RandomForestClassifier(random_state=42)

# Setting up grid searches
# param_grid_svm = [
#     {'kernel':['rbf','linear','poly'],'C':[0.1,1,10,100,1000],'degree':[3,4,5,10]}]

# Result of parameter search - Turn on if you want to skip param search
param_grid_svm = [
    {'C': [0.1], 'degree': [3], 'kernel': ['poly']}]

# param_grid_tree = [
#     {'max_depth':[1,3,5,10,15,20,None],'min_samples_split':[2,5,10,20,40],
#      'min_samples_leaf':[1,5,10],'criterion':['gini','entropy']}]
   
# Result of parameter search - Turn on if you want to skip param search
param_grid_tree = [
    {'criterion': ['entropy'], 'max_depth': [10], 'min_samples_leaf':[ 1], 'min_samples_split': [5]}]
 
# param_grid_rfc = [
#     {'n_estimators':[100,200,400,600,800],'max_depth':[1,3,5,10,20,40],'min_samples_leaf':[1,5,10],
#       'max_leaf_nodes': [None,8,16,32,64]}]   

# Result of parameter search - Turn on if you want to skip param search - ADVISE TURNING ON AS THIS TAKES A WHILE. 
param_grid_rfc = [
    {'max_depth': [5], 'max_leaf_nodes': [16], 'min_samples_leaf': [1], 'n_estimators': [400]}]  

# SVM
grid_search_svm = GridSearchCV(svm_clf, param_grid_svm, cv=5, scoring = 'accuracy', return_train_score = True)
grid_search_svm.fit(train_features_prep,target_features)
print('SVM Done!')

# Decision Tree
grid_search_tree = GridSearchCV(tree_clf, param_grid_tree, cv=5, scoring = 'accuracy', return_train_score = True)
grid_search_tree.fit(train_features_prep,target_features)
print('Decision Tree Done!')

# Random Forest
grid_search_rfc = GridSearchCV(rfc, param_grid_rfc, cv=5, scoring = 'accuracy', return_train_score = True)
grid_search_rfc.fit(train_features_prep,target_features)
print('Random Forest Done!')
    
# Print Best Results
print('\n') 
print('SVM Best Params: ' , grid_search_svm.best_params_)
print('SVM Best Score: ' , grid_search_svm.best_score_ )
print('\n') 
print('Tree Best Params: ' , grid_search_tree.best_params_)
print('Tree Best Score: ' , grid_search_tree.best_score_ )
print('\n') 
print('RFC Best Params: ' , grid_search_rfc.best_params_)
print('RFC Best Score: ' , grid_search_rfc.best_score_ )


# -------------------------- Save Best Tree as a png -------------------------------------

export_graphviz( 
    grid_search_tree.best_estimator_,out_file='Figures/Decision_Tree_Diagrams/Submission_7_Tree.dot',
    feature_names = post_transform_train_features,
    class_names = 'Survived',
    rounded=True,
    filled=True
    )

(graph,) = pydot.graph_from_dot_file('Figures/Decision_Tree_Diagrams/Submission_7_Tree.dot')
graph.write_png('Figures/Decision_Tree_Diagrams/Submission_7_Tree.png')


# --------------- Plotting Learning Curve: Submission 6 only  -----------------

fig,ax = plt.subplots(figsize=(12,8))
ax.set_ylim(0.6,1.02)
ax.set_xlim(0,720)
ax.set_title('Submission 6 - SVM')
mlib.plot_learning_curve(
    grid_search_svm.best_estimator_, train_features_prep, target_features, 'accuracy', 5, ax)



# --------------- Plotting Learning Curve: Submission 7 only  -----------------

fig,ax = plt.subplots(figsize=(12,8))
ax.set_ylim(0.6,1.02)
ax.set_xlim(0,720)
ax.set_title('Submission 7 - Decision Tree')
mlib.plot_learning_curve(
    grid_search_tree.best_estimator_, train_features_prep, target_features, 'accuracy', 5, ax)



# --------------- Plotting Learning Curve: Submission 8 only  -----------------

fig,ax = plt.subplots(figsize=(12,8))
ax.set_ylim(0.6,1.02)
ax.set_xlim(0,720)
ax.set_title('Submission 8 - Random Forest')
mlib.plot_learning_curve(
    grid_search_rfc.best_estimator_, train_features_prep, target_features, 'accuracy', 5, ax)



# --------------- Plotting Learning Curve: Comparison  -----------------

fig,[ax1,ax2,ax3] = plt.subplots(1,3,sharex=True, sharey = True,figsize=(12,8))
ax1.set_ylim(0.6,1.02)
ax1.set_xlim(0,720)
ax1.set_title('SVM')
ax2.set_title('Decision Tree')
ax3.set_title('Random Forest')
mlib.plot_learning_curve(
    grid_search_svm.best_estimator_, train_features_prep, target_features, 'accuracy', 5, ax1)
mlib.plot_learning_curve(
    grid_search_tree.best_estimator_, train_features_prep, target_features, 'accuracy', 5, ax2)
mlib.plot_learning_curve(
    grid_search_rfc.best_estimator_, train_features_prep, target_features, 'accuracy', 5, ax3)



# --------------- Predicting on test data and save submission -------------------------

# # Submission 6 - SVM
# svm_predictions=grid_search_svm.best_estimator_.predict(test_features_prep)
# mlib.Pred_to_Kaggle_Format(svm_predictions,'Submissions/Submission_6.csv')

# # Submission 7 - Decision Tree
# tree_predictions=grid_search_tree.best_estimator_.predict(test_features_prep)
# mlib.Pred_to_Kaggle_Format(tree_predictions,'Submissions/Submission_7.csv')

# # Submission 8 - Random Forest
# rfc_predictions=grid_search_rfc.best_estimator_.predict(test_features_prep)
# mlib.Pred_to_Kaggle_Format(rfc_predictions,'Submissions/Submission_8.csv')




# ------------------------ Finish --------------------------------------------
    
print('\n') 
print('Script runtime:', (time.time()-start)/60)
print('\n' '\n')
print('----------------- End --------------------')