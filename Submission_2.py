# -*- coding: utf-8 -*-
"""
Created on Thu May 14 17:54:37 2020

Kaggle Titanic: Machine Learning for a Disaster
More information is here: https://www.kaggle.com/c/titanic/data

Document Objective: Submission_2


@author: edwin
"""

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
print('\n')
print('--------------- Start --------------------')
print('\n' '\n')
start=time.time()

# -------------------- Executing Pipeline on training data -------------------

# Prep data for Pipes - Training Data
path = 'Original_Data/train.csv'
train_original = mlib.csv_to_df(path)
feature_list = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']
target_list = ['Survived']

# Define Numeric Pipeline - Standard Scaling
num_pipe = Pipeline([
    ('imputer_mean', SimpleImputer(strategy='mean')),
    #('std_scalar', StandardScaler())
])

# Define Categorical Pipeline - One Hot Enconding
cat_pipe = Pipeline([
    ('imputer' , SimpleImputer(strategy = 'most_frequent')),
    #('ohe' , OneHotEncoder()),
    ('oe', OrdinalEncoder())
    
])

#Combinin Pipes into full pipeline 
full_pipeline,train_features,target_features, post_transform_train_features = mlib.Full_PipeLine(path,feature_list,target_list,num_pipe, cat_pipe)

# Transform data using final combined pipeline
train_features_prep = full_pipeline.fit_transform(train_features)

# -------------------- Executing Pipeline on test data -----------------------

# Prep data for Pipes - Test Data
path = 'Original_Data/test.csv'
test_original = mlib.csv_to_df(path)
target_list = []
full_pipeline,test_features,empty,post_transform_test_features = mlib.Full_PipeLine(path,feature_list,target_list,num_pipe, cat_pipe)

# Transform data using final combined pipeline
test_features_prep = full_pipeline.fit_transform(test_features)

# -------------------------- Decision Tree  ------------------------------

# Decision Tree
tree_clf = DecisionTreeClassifier()

# Cross Validation if desired
## cvs = cross_val_score(tree_clf, train_features_prep,target_features,cv =5,scoring='accuracy')
## print(sorted(cvs,reverse=True))

# ---------- Grid Search ------------
param_grid = [
    {'max_depth':[1,3,5,10,None],'min_samples_split':[2,5,10,20,40],'min_samples_leaf':[1,5,10,20],'criterion':['gini','entropy']}
    ]

grid_search = GridSearchCV(tree_clf, param_grid, cv=5, scoring = 'accuracy', return_train_score = True)
grid_search.fit(train_features_prep,target_features)

#Output scores for each combination of parameters
cvres = grid_search.cv_results_
for mean_score,params in zip(cvres['mean_test_score'],cvres[ 'params']):
    print(mean_score , params)

# ---------- Save Best Tree as a png ---------------
export_graphviz( 
    grid_search.best_estimator_,out_file='tree_test.dot',
    feature_names = post_transform_train_features,
    class_names = 'Survived',
    rounded=True,
    filled=True
    )

print('\n') 
print('Best Params: ' , grid_search.best_params_)
print('Best Score: ' , grid_search.best_score_ )


(graph,) = pydot.graph_from_dot_file('tree_test.dot')
graph.write_png('tree_test.png')

# --------------- Plotting Learning Curve for best result -----------------
fig,ax = plt.subplots()
mlib.plot_learning_curve(grid_search.best_estimator_, train_features_prep, target_features, 'accuracy', 5, ax)   

# --------------- Predicting on test data -------------------------

predictions=grid_search.best_estimator_.predict(test_features_prep)
mlib.Pred_to_Kaggle_Format(predictions,'Submissions/Submission_2.csv')

# ------------------------ Finish --------------------------------------------
    
print('\n') 
print('Script runtime:', (time.time()-start)/60)
print('\n' '\n')
print('----------------- End --------------------')