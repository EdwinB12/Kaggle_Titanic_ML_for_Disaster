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
    ('std_scalar', StandardScaler())
])

# Define Categorical Pipeline - One Hot Enconding
cat_pipe = Pipeline([
    ('imputer' , SimpleImputer(strategy = 'most_frequent')),
    ('ohe' , OneHotEncoder())
])

#Combinin Pipes into full pipeline 
full_pipeline,train_features,target_features,post_trans_train_feature = mlib.Full_PipeLine(path,feature_list,target_list,num_pipe, cat_pipe)

# Transform data using final combined pipeline
train_features_prep = full_pipeline.fit_transform(train_features)



# -------------------- Executing Pipeline on test data -----------------------

# Prep data for Pipes - Test Data
path = 'Original_Data/test.csv'
test_original = mlib.csv_to_df(path)
target_list = []
full_pipeline,test_features,empty,post_trans_test_feature = mlib.Full_PipeLine(path,feature_list,target_list,num_pipe, cat_pipe)

# Transform data using final combined pipeline
test_features_prep = full_pipeline.fit_transform(test_features)



# -------------------------- SVM classifier rbf ------------------------------

# Initialise SVM parameters
svm_clf_rbf_C100 = SVC(kernel='rbf',C=100)

# Fit SVM to data
svm_clf_rbf_C100.fit(train_features_prep,target_features)



# -------------------------- Cross Validation ------------------------------

# 10 fold cross validation and print scores
cvs = cross_val_score(svm_clf_rbf_C100,train_features_prep,target_features ,cv=5)
print(sorted(cvs,reverse=True))



# ----------------- Plotting Learning Curve for best result -------------------

# Plotting Learning Curve
fig,ax = plt.subplots()
ax.set_title('Submission 1 - SVM Learning Curve')
ax.set_ylim(0.6,1.02)
ax.set_xlim(0,720)
mlib.plot_learning_curve(svm_clf_rbf_C100, train_features_prep, target_features, 'accuracy', 5, ax)



# ---------------- Predicting on test data and save submission ---------------

# Predict on prepared test dataset and write out to csv
predictions=svm_clf_rbf_C100.predict(test_features_prep)
mlib.Pred_to_Kaggle_Format(predictions,'Submissions/Submission_1.csv')



# ------------------------ Finish --------------------------------------------
    
print('\n') 
print('Script runtime:', (time.time()-start)/60)
print('\n' '\n')
print('----------------- End --------------------')