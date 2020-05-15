# -*- coding: utf-8 -*-
"""
Created on Wed May 13 15:17:17 2020

Kaggle Titanic: Machine Learning for a Disaster
More information is here: https://www.kaggle.com/c/titanic/data

Document Objective: Parameter searching and optimising

@author: edwin
"""
#%% 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import time
import pickle
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import cross_val_score, learning_curve, GridSearchCV
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC,LinearSVC 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


print('\n')
print('--------------- Start --------------------')
print('\n' '\n')
start=time.time()

#%% 

# Loading in S2 dataframe - no feature scaling
with open('C:\\Users\\edwin\\OneDrive\\Documents\\Machine_Learning\\Git_Repos\\Kaggle_Titanic_ML_for_Disaster\\Pickled_Files\\train_df_S2.txt', 'rb') as myFile:
    train_df_S2 = pickle.load(myFile)
    
# Assigning training and target features to seperate numpy arrays
target_feature = train_df_S2['Survived'].to_numpy()
training_features = train_df_S2[['Sex','Pclass','Fare','Embarked','Parch','SibSp']].to_numpy()

# Feature scale training features
scalar = StandardScaler()
training_features_SS = scalar.fit_transform(training_features)

svm_clf = SVC()

param_grid = [
    {'C':[0.1,1,10,100,500,1000],'kernel':['linear', 'poly','rbf']}]

grid_search = GridSearchCV(svm_clf, param_grid, cv=10, scoring = 'accuracy', return_train_score = True)
grid_search.fit(training_features_SS,target_feature)

#Output scores for each combination of parameters
cvres = grid_search.cv_results_
for mean_score,params in zip(cvres['mean_test_score'],cvres[ 'params']):
    print(mean_score , params)
    

#%% Predict with optimised hyperparameters
    
# Train SVM
svm_clf_opt = SVC(kernel = 'rbf', C = 100, random_state=42)
svm_clf_opt.fit(training_features,target_feature)
    




#%% Finish
    
print('\n') 
print('Script runtime:', (time.time()-start)/60)
print('\n' '\n')
print('----------------- End --------------------')