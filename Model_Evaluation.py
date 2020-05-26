# -*- coding: utf-8 -*-
"""
Created on Mon May 25 15:41:38 2020

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
from sklearn.model_selection import cross_val_score, learning_curve, GridSearchCV,train_test_split , cross_val_predict
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier,GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, precision_score,recall_score,f1_score, precision_recall_curve, roc_curve, roc_auc_score, plot_confusion_matrix
# Custom 
import MLFunLib as mlib # Custom made library


print('\n')
print('--------------- Start --------------------')
print('\n' '\n')
start=time.time()

# ----------------------------------- PREP 1 ----------------------------------------------------
# -----------------------------------------------------------------------------------------------
# --------------- Data Read, Feature Engineering and data prep for training ---------------------

# Read Train and Test Datasets and save off original copies
train_path = 'Original_Data/train.csv'
train_original = pd.read_csv(train_path)
train_df = train_original.copy()

# Feature Engineering 
train_df = mlib.Feature_Engineering(train_df)

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
full_pipeline_train,train_features1,train_target, post_transform_train_features1 = mlib.Full_PipeLine(
    train_df,feature_list,target_list,num_pipe, cat_pipe)

# Transform data using final combined pipeline
train_features_prep1 = full_pipeline_train.fit_transform(train_features1)


# ----------------------------------- PREP 2 ----------------------------------------------------
# -----------------------------------------------------------------------------------------------
# --------------- Data Read, Feature Engineering and data prep for training ---------------------

# Read Train and Test Datasets and save off original copies
train_path = 'Original_Data/train.csv'
train_original = pd.read_csv(train_path)
train_df = train_original.copy()


# Feature Engineering 
train_df = mlib.Feature_Engineering(train_df)

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
full_pipeline_train,train_features2,target_features, post_transform_train_features2 = mlib.Full_PipeLine(
    train_df,feature_list,target_list,num_pipe, cat_pipe)

# Transform data using final combined pipeline
train_features_prep2 = full_pipeline_train.fit_transform(train_features2)




# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------

# ----------------------------- Defining models  ------------------------------

# Submission 1
sub1 = SVC(kernel='rbf',C=100,random_state=42)
sub1_prob = SVC(kernel='rbf',C=100,random_state=42,probability=True) #SVM needs a version with prob set to true

# Submission 2
sub2 = DecisionTreeClassifier(criterion='gini',max_depth=None, min_samples_leaf = 5,
                                  min_samples_split = 2, random_state = 42)

# Submission 5
sub5 = RandomForestClassifier(max_depth=10, max_leaf_nodes = 32, 
                              min_samples_leaf = 1, n_estimators =200,random_state=42)

# Submission 6
sub6 = SVC(C=0.1,degree=3,kernel='poly',random_state=42) 
sub6_proba = SVC(C=0.1,degree=3,kernel='poly',random_state=42,probability=True) #SVM needs a version with prob set to true

# Submission 7
sub7 = DecisionTreeClassifier(criterion='entropy',max_depth= 10,
                                  min_samples_leaf=1,min_samples_split=5,random_state=42) 

# Submission 8
sub8 = RandomForestClassifier(max_depth=5,max_leaf_nodes=16,
                              min_samples_leaf=1,n_estimators=400,random_state=42)

sub9 = AdaBoostClassifier(
    DecisionTreeClassifier(random_state=42,max_depth = 5,
                             max_leaf_nodes = 16,min_samples_leaf=1),
    random_state=42,algorithm='SAMME.R',learning_rate=0.01,n_estimators=800
    )

sub10 = GradientBoostingClassifier(
    random_state=42,max_depth = 5, max_leaf_nodes = 16,min_samples_leaf=1,n_estimators=800,
    learning_rate = 0.001, subsample = 1)


# ---------------------------------------------------------------------------------------------
# ------------------------------ Evaluation Functions ---------------------------------------------------

# Calculating Precision, Recall, F1 Score and Confusion Matrix plot
def Class_Metrics_SepPlot(classifier,train_features,train_labels,sub_name): 
    
    # Cross validation predictions
    cv_predictions =cross_val_predict(classifier,train_features,train_labels,cv=5) 
    
    # Create confusion matrix, precision and recall metrics using cross validated predictions 
    conf_matrix = confusion_matrix(train_target, cv_predictions)
    conf_matrix_norm = confusion_matrix(train_target, cv_predictions,normalize='true')
    prec = precision_score(train_target, cv_predictions)
    recall = recall_score(train_target, cv_predictions)
    f1 = f1_score(train_target,cv_predictions)

    # Printing Key Statistics
    print('\n')
    print('Precision:', prec)
    print("%.2f" % (prec*100),'% of positive predictions are correct.')
    print('\n')
    print('Recall:', recall)
    print("%.2f" % (recall*100),'% of people who survived are detected.')
    print('\n')
    print('F1 Score:', f1)
    
    # Plot correlation matrix
    fig,[ax1,ax2] = plt.subplots(1,2,figsize=(8,4))
    sns.heatmap(conf_matrix,annot=True,fmt="d",ax=ax1,square=True,cmap='coolwarm',cbar=False)
    sns.heatmap(conf_matrix_norm,annot=True,fmt=".2f",ax=ax2,square=True,cmap='coolwarm',cbar=False )
    ax1.set_title('Non-Normalised')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Real')
    ax2.set_title('Normalised')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Real')
    fig.suptitle('Confusion Matrix: ' + sub_name )
    
    return prec,recall,f1

# Plotting Precision, Recall and ROC Curves
def Prob_Metrics_SepPlot(classifier,train_features,train_labels,method,sub_name):

    # Cross validation predictions
    cv_predictions =cross_val_predict(classifier,train_features,
                                      train_labels,cv=5,method=method) 

    # Precision recall curves
    precisions,recalls,thresholds = precision_recall_curve(train_labels, cv_predictions[:,1]) 

    # Plotting precision vs recall plots
    fig,[ax1,ax2] = plt.subplots(1,2,figsize=(12,6))
    ax1.plot((0.5,0.5),(0,1),'r--')
    ax1.plot(thresholds,precisions[:-1],'b-',label='Precision')
    ax1.plot(thresholds,recalls[:-1],'g-',label='Recall')
    ax2.plot(recalls,precisions)
    ax1.set_title('Precision and Recall vs Threshold')
    ax2.set_title('Recall vs Precision')
    ax1.set_xlabel('Threshold')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax1.set_xlim(0,1)
    ax1.set_ylim(0,1)
    ax2.set_xlim(0,1)
    ax2.set_ylim(0,1)
    ax1.legend()
    ax1.grid(True)
    ax2.grid(True)
    fig.suptitle(sub_name)
    
    # ROC Curve Calculation
    fpr,tpr,thresholds = roc_curve(train_target, cv_predictions[:,1]) #
    
    # Plot ROC Curve
    fig,ax = plt.subplots(figsize=(6,6))
    ax.plot(fpr,tpr)
    ax.plot((0,1),(0,1),'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve: ' + sub_name)
    ax.grid(True)

    # Calculate Area Under ROC Curve
    roc_auc = roc_auc_score(train_target, cv_predictions[:,1])
    print('\n')
    print('Area under ROC Curve:', roc_auc)
    
    # Add to plot
    ax.text(0.51,0.1,'Area under ROC Curve:' "%.2f" % roc_auc,
            bbox=dict(facecolor='white', alpha=0.5))
    
# Function to plot single ROC Curves onto plots
def ROC_Curve_Plot(classifier,train_features,train_labels,method,sub_name,ax,fmt):   
   
    # Cross validation predictions
    cv_predictions =cross_val_predict(classifier,train_features,
                                      train_labels,cv=5,method=method) 

    # ROC Curve Calculation
    fpr,tpr,thresholds = roc_curve(train_target, cv_predictions[:,1]) #
    
    # Plot ROC Curve
    ax.plot(fpr,tpr,fmt,label=sub_name)

    # Calculate Area Under ROC Curve
    roc_auc = roc_auc_score(train_target, cv_predictions[:,1])
    print('\n')
    print('Area under ROC Curve - ',sub_name, ':', roc_auc)
    
    return roc_auc




# ------------------------------ Evaluation Plots ----------------------------------------

print('----------------------- Submission 1 --------------------------')
Class_Metrics_SepPlot(sub1, train_features_prep1, target_features, 'Submission 1')
Prob_Metrics_SepPlot(sub1_prob,train_features_prep1,target_features,'predict_proba','Submission 1') 
print('\n')

print('----------------------- Submission 2 --------------------------')
Class_Metrics_SepPlot(sub2, train_features_prep1, target_features, 'Submission 2')
Prob_Metrics_SepPlot(sub2,train_features_prep1,target_features,'predict_proba','Submission 2')
print('\n')

print('----------------------- Submission 5 --------------------------')
Class_Metrics_SepPlot(sub5, train_features_prep1, target_features, 'Submission 5')
Prob_Metrics_SepPlot(sub5,train_features_prep1,target_features,'predict_proba','Submission 5')
print('\n')

print('----------------------- Submission 6 --------------------------')
Class_Metrics_SepPlot(sub6, train_features_prep2, target_features, 'Submission 6')
Prob_Metrics_SepPlot(sub6_proba,train_features_prep2,target_features,'predict_proba','Submission 6') 
print('\n')

print('----------------------- Submission 7 --------------------------')
Class_Metrics_SepPlot(sub7, train_features_prep2, target_features, 'Submission 7')
Prob_Metrics_SepPlot(sub7,train_features_prep2,target_features,'predict_proba','Submission 7')
print('\n')

print('----------------------- Submission 8 --------------------------')
Class_Metrics_SepPlot(sub8, train_features_prep2, target_features, 'Submission 8')
Prob_Metrics_SepPlot(sub8,train_features_prep2,target_features,'predict_proba','Submission 8')
print('\n') 

print('----------------------- Submission 9 --------------------------')
Class_Metrics_SepPlot(sub9, train_features_prep2, target_features, 'Submission 9')
Prob_Metrics_SepPlot(sub9,train_features_prep2,target_features,'predict_proba','Submission 9')
print('\n')

print('----------------------- Submission 10 --------------------------')
Class_Metrics_SepPlot(sub10, train_features_prep2, target_features, 'Submission 10')
Prob_Metrics_SepPlot(sub10,train_features_prep2,target_features,'predict_proba','Submission 10')
print('\n')

# Multiple ROC Curves on a single axis

print('----------------------- ROC Curve - Random Forest --------------------------')
fig,ax = plt.subplots()
ROC_Curve_Plot(sub5, train_features_prep1, target_features,'predict_proba', 'Submission 5',ax,'g-')
ROC_Curve_Plot(sub8, train_features_prep2, target_features,'predict_proba', 'Submission 8',ax,'b-')
ax.legend()
ax.set_xlabel('False Positive Rate') 
ax.set_ylabel('True Positive Rate')   
ax.set_title('Random Forests')
    
print('----------------------- ROC Curve -  Decision Tree --------------------------')
fig,ax = plt.subplots()
ROC_Curve_Plot(sub2, train_features_prep1, target_features,'predict_proba', 'Submission 2',ax,'g-')
ROC_Curve_Plot(sub7, train_features_prep2, target_features,'predict_proba', 'Submission 7',ax,'b-')
ax.legend()
ax.set_xlabel('False Positive Rate') 
ax.set_ylabel('True Positive Rate')   
ax.set_title('Decision Trees')

print('----------------------- ROC Curve -  SVM --------------------------')
fig,ax = plt.subplots()
ROC_Curve_Plot(sub1_prob, train_features_prep1, target_features,'predict_proba', 'Submission 1',ax,'g-')
ROC_Curve_Plot(sub6_proba, train_features_prep2, target_features,'predict_proba', 'Submission 6',ax,'b-')
ax.legend()
ax.set_xlabel('False Positive Rate') 
ax.set_ylabel('True Positive Rate')   
ax.set_title('SVM')

print('----------------------- ROC Curve - Prep1 --------------------------')
fig,ax = plt.subplots()
ROC_Curve_Plot(sub1_prob, train_features_prep1, target_features,'predict_proba', 'Submission 1',ax,'g-')
ROC_Curve_Plot(sub2, train_features_prep1, target_features,'predict_proba', 'Submission 2',ax,'b-')
ROC_Curve_Plot(sub5, train_features_prep1, target_features,'predict_proba', 'Submission 5',ax,'m-')
ax.legend()
ax.set_xlabel('False Positive Rate') 
ax.set_ylabel('True Positive Rate')   
ax.set_title('Prep 1')

print('----------------------- ROC Curve - Prep2 --------------------------')
fig,ax = plt.subplots()
ROC_Curve_Plot(sub6_proba, train_features_prep2, target_features,'predict_proba', 'Submission 6',ax,'g-')
ROC_Curve_Plot(sub7, train_features_prep2, target_features,'predict_proba', 'Submission 7',ax,'b-')
ROC_Curve_Plot(sub8, train_features_prep2, target_features,'predict_proba', 'Submission 8',ax,'m-')
ax.legend()
ax.set_xlabel('False Positive Rate') 
ax.set_ylabel('True Positive Rate')   
ax.set_title('Prep 2')

