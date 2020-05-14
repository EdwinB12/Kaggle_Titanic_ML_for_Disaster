# -*- coding: utf-8 -*-
"""
Created on Thu May 14 16:56:34 2020

Kaggle Titanic: Machine Learning for a Disaster
More information is here: https://www.kaggle.com/c/titanic/data

Document Objective: Importable Functions for the Kaggle Titanic: Machine Learning for a Disaster challenge. 

@author: edwin
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import learning_curve

# -------------------------------- Read csv files to dfs ------------------------------------------

def csv_to_df(path): 
    df = pd.read_csv(path)
    return df

#--------------------------------------------------------------------------------------------------

# -------------------------------- Plot Learning Curve --------------------------------------------

def plot_learning_curve(estimator,X,y,scoring,fold,ax): 
    
    
    
    '''
    Function to plot learning curves for given estimator (e.g. SGD classifier).
    Cross validation is performed with fold 10 and 50 linearly spaced number of training
    examples (e.g. linspace(0.1,1,50)). 
    

    Parameters
    ----------
    estimator : object
        Method of estimation (e.g. SGD classifier, SVM, Decision Tree etc.)
    X : Training Features
        Features provided to train estimator
    y : Target Feature
        Feature to be predicted
    scoring: metric of choice
        metrics can be 'accuracy', 'loss' etc. Must be a string! 
    fold: int
        cross validation parameter (cv)
    ax : Axis
        Axis for plotting

    Returns
    -------
    None.

    '''
    
    # Calculating train and val accuracy with 50 different training set sizes 
    train_size,train_scores,valid_scores = learning_curve(
        estimator,X,y,cv=fold,random_state=42,
        train_sizes=np.linspace(0.01, 1.0, 50),scoring=scoring)
    
    # Calculating mean and standard deviations for training and val sets
    train_mean = np.mean(train_scores,axis=1)
    train_std = np.std(train_scores,axis=1)
    valid_mean = np.mean(valid_scores,axis=1)
    valid_std = np.std(valid_scores,axis=1)
    
    # Plotting Lines
    ax.plot(train_size, train_mean, color="royalblue",  label="Training score")
    ax.plot(train_size, valid_mean, '--', color="#111111", label="Cross-validation score")

    # Draw error bands
    #ax.fill_between(train_size, train_mean - train_std, train_mean + train_std, color="blueviolet")
    ax.fill_between(train_size, valid_mean - valid_std, valid_mean + valid_std, color="#DDDDDD")
    
    #Plotting details
    ax.set_xlabel("Training Set Size") 
    ax.set_ylabel("Accuracy Score") 
    ax.legend(loc="lower right")

#--------------------------------------------------------------------------------------------------

# -------------------------------- Combining Full PipeLine ----------------------------------------

def Full_PipeLine(path, feature_list, target_list,  num_pipe, cat_pipe):
    
    # Read in data
    df = csv_to_df(path)
    
    # Split df into training and target features. 
    train_features = df[feature_list]
    target_features = df[target_list].to_numpy().ravel()
    
    #Assigning feature labels to numeric and categoric based on dtypes
    numeric_features = train_features.select_dtypes(include=['int64','float64']).columns
    categoric_features = train_features.select_dtypes(include=['object']).columns

    # Combining seperate pipes into full pipeline
    full_pipeline = ColumnTransformer([
        ('num' , num_pipe,numeric_features), # Apply numeric pipeline to numeric features
        ('cat', cat_pipe, categoric_features), # Apply categoric pipeline to numeric features
    ])
    
    return full_pipeline, train_features, target_features

#--------------------------------------------------------------------------------------------------
# -------------------------------- Saved CSV for Kaggle Submission -----------------------------------------

# Function preparing predictons for Kaggle Submission 
def Pred_to_Kaggle_Format(predictions,csv_path):

    df = pd.read_csv('Original_Data/test.csv')
    pass_id = df['PassengerId'].to_numpy().ravel()
    pred_df = pd.DataFrame({'PassengerId' : pass_id, 'Survived' : predictions})
    #pred_df['Survived'] = predictions
    pred_df.to_csv(csv_path,index=False)

#---------------------------------------------------------------------------------------------------
    
    
    
    