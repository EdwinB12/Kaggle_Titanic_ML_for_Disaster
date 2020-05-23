# -*- coding: utf-8 -*-
"""
Created on Thu May 21 22:23:04 2020


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


# Put together data and Print P correlations
corr_df = pd.DataFrame(data = train_features_prep, columns = post_transform_test_features)
corr_df['Survived'] = target_features
corr_matrix = corr_df.corr()
print(corr_matrix['Survived'].sort_values(ascending=False))


# -------------------------------------------------------------------------------------------------
# -------------------------------- Decision Tree and Random Forest --------------------------------

# Initialise training methods for boosting. Using optimised parameters from Submissions 6,7,8

# Decision Tree
tree_clf = DecisionTreeClassifier(random_state=42,criterion='entropy',max_depth=10,
                                  min_samples_leaf=1,min_samples_split=5) 

# Decision Tree cross validation score 
tree_clf.fit(train_features_prep,target_features)
print('\n','tree_clf CVS score:' , np.mean(cross_val_score(tree_clf, train_features_prep,target_features,cv=5,
                scoring = 'accuracy')))
print('Decision Tree Done!','\n')

# Random Forest
rfc = RandomForestClassifier(random_state=42,max_depth = 5,
                             max_leaf_nodes = 16,min_samples_leaf=1,n_estimators=400)

# Random Forest cross validation score
rfc.fit(train_features_prep,target_features)
print('rfc CVS score:' , np.mean(cross_val_score(rfc, train_features_prep,target_features,cv=5,
                scoring = 'accuracy')))
print('Random Forest Done!','\n')

# Output features and their respective importance
print('Feature Importance')
for name,score in zip(post_transform_train_features,rfc.feature_importances_):
    print(name,score)
    


#---------------------------------- Ada Boosting ---------------------------------------------------------

# AdaBoost classifier with decision trees (decision tree params same as random forest)
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(random_state=42,max_depth = 5,
                             max_leaf_nodes = 16,min_samples_leaf=1),
    random_state=42
    )

# Parameter grid used for original parameter search
# param_grid_ada = [
#     {'n_estimators':[400,600,800,1000], 'algorithm':['SAMME.R'],
#      'learning_rate':[0.001,0.01,0.1,0.5,1,10]}]

# Param grid with best parameters
param_grid_ada_best = [
    {'algorithm': ['SAMME.R'], 'learning_rate': [0.01], 'n_estimators': [800]}
    ]

# AdaBoost Gridsearch 
grid_Search_ada_best = GridSearchCV(ada_clf, param_grid_ada_best, cv=5, scoring = 'accuracy', return_train_score = True)
grid_Search_ada_best.fit(train_features_prep,target_features)

# Output best Cross Validation score and parameters from grid search
print('\n') 
print('Ada Best Params: ' , grid_Search_ada_best.best_params_) # {'algorithm': 'SAMME.R', 'learning_rate': 0.01, 'n_estimators': 800}
print('Ada Best Score: ' , grid_Search_ada_best.best_score_ ) # 0.8159814198732033
print('Ada Done!')



# ---------------------------------- Gradient Boosting ---------------------------------------------------------

grad_clf = GradientBoostingClassifier(
    random_state=42,max_depth = 5, max_leaf_nodes = 16,min_samples_leaf=1)

# param_grid_grad = [
#     {'n_estimators':[400,800,1200,1600],'learning_rate':[0.0001,0.005,0.001,0.01],'subsample':[0.25,0.5,1]}]

param_grid_grad_best = [
    {'n_estimators':[800],'learning_rate':[0.001],
     'subsample':[1]}
    ]

# Gradient Boost Gridsearch 
grid_search_grad_best = GridSearchCV(grad_clf, param_grid_grad_best, cv=5, scoring = 'accuracy', return_train_score = True)
grid_search_grad_best.fit(train_features_prep,target_features)

#Output best Cross Validation score and parameters from grid search
print('\n') 
print('Gradient Best Params: ' , grid_search_grad_best.best_params_)
print('Gradient Best Score: ' , grid_search_grad_best.best_score_ )
print('Gradient Boosting Done!')




# --------------- Plotting Learning Curve: Ada Boost (Submission 9) only -----------------
fig,ax = plt.subplots(figsize=(12,8))
ax.set_ylim(0.6,1.02)
ax.set_xlim(0,720)
ax.set_title('Submission 9: Decision Tree - Ada Boost')
mlib.plot_learning_curve(
    grid_Search_ada_best.best_estimator_, train_features_prep, target_features, 'accuracy', 5, ax)




# --------------- Plotting Learning Curve: Gradient Boost (Submission 10) only -----------------
fig,ax = plt.subplots(figsize=(12,8))
ax.set_ylim(0.6,1.02)
ax.set_xlim(0,720)
ax.set_title('Submission 10: Decision Tree - Gradient Boost')
mlib.plot_learning_curve(
    grid_search_grad_best.best_estimator_, train_features_prep, target_features, 'accuracy', 5, ax)





# --------------- Plotting Learning Curve Comparison - This may take a while and is hence turned off -----------------

# fig3,[[ax1,ax2],[ax3,ax4]] = plt.subplots(2,2,sharex=True, sharey = True,figsize=(12,8))
# ax1.set_ylim(0.6,1.02)
# ax1.set_xlim(0,720)
# ax1.set_title('Decision Tree')
# ax2.set_title('Random Forest')
# ax3.set_title('Decision Tree - Ada Boosting')
# ax4.set_title('Decision Tree - Gradient Boosting')
# mlib.plot_learning_curve(
#     tree_clf, train_features_prep, target_features, 'accuracy', 5, ax1)
# mlib.plot_learning_curve(
#     rfc, train_features_prep, target_features, 'accuracy', 5, ax2)
# mlib.plot_learning_curve(
#     grid_Search_ada_best.best_estimator_, train_features_prep, target_features, 'accuracy', 5, ax3)
# mlib.plot_learning_curve(
#     grid_search_grad_best.best_estimator_, train_features_prep, target_features, 'accuracy', 5, ax4)



#--------------- Predicting on test data and save submission -------------------------

# # Submission 9 - Ada Boost
# ada_predictions=grid_Search_ada_best.best_estimator_.predict(test_features_prep)
# mlib.Pred_to_Kaggle_Format(ada_predictions,'Submissions/Submission_9.csv')

# # Submission 10 - Gradient Boost
# grad_predictions=grid_search_grad_best.best_estimator_.predict(test_features_prep)
# mlib.Pred_to_Kaggle_Format(grad_predictions,'Submissions/Submission_10.csv')



# ------------------------ Finish --------------------------------------------
    
print('\n') 
print('Script runtime:', (time.time()-start)/60)
print('\n' '\n')
print('----------------- End --------------------')