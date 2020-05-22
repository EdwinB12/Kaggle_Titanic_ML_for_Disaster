# -*- coding: utf-8 -*-
"""
Created on Sat May 16 14:09:32 2020

Kaggle Titanic: Machine Learning for a Disaster
More information is here: https://www.kaggle.com/c/titanic/data

Document Objective: Submission_3 and Submission_4

Workflow: 
    1. Choose six features from training data:'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked'
    2. Made two seperate pipes: 
        - Numerical Pipe: Replace missing values with mean values (1 value only in the test dataset), Standard feature scaling
        - Categorical Pipe: Replace missing values with most common (2 instances for 'Embarked' class); Ordinal Encoding
    3. Combine pipes into a full pipeline and push training and test data through. 
    4. Initiliase models: 
        SVM from Submission 1 with respective parameters. 
        Decision Tree from Submission 2 with respective parameters.
        Ensemble Voting CLassifier - Hard Voting.
        Ensemble Voting Classifier - Soft Voting.
    5. Fit the two voting classifiers for predictions.
    6. 5 Fold Cross Validation and print models with respective scores. 
    7. Plot learning curve of two ensemble methods 
    8. Write both Hard and Soft voting  strategies out to CSV file
    9. Submit to Kaggle as: 
        Submission 3 - Ensemble Hard Voting
        Submission 4 - Ensemble Soft Voting 
    
Things to try for future submissions: 
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
from sklearn.ensemble import VotingClassifier

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
    #('ohe' , OneHotEncoder()),
    ('oe', OrdinalEncoder())
    
])

#Combinin Pipes into full pipeline 
full_pipeline,train_features,target_features, post_transform_train_features = mlib.Full_PipeLine(
    train_original,feature_list,target_list,num_pipe, cat_pipe)

# Transform data using final combined pipeline
train_features_prep = full_pipeline.fit_transform(train_features)



# -------------------- Executing Pipeline on test data -----------------------

# Prep data for Pipes - Test Data
path = 'Original_Data/test.csv'
test_original = mlib.csv_to_df(path)
target_list = []
full_pipeline,test_features,empty,post_transform_test_features = mlib.Full_PipeLine(
    test_original,feature_list,target_list,num_pipe, cat_pipe)

# Transform data using final combined pipeline
test_features_prep = full_pipeline.fit_transform(test_features)





# ------------------ Training Different Estimators --------------------

# Support Vector Classifier - Submission 1
svc_clf = SVC(C=100, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
    max_iter=-1, probability=False, random_state=42, shrinking=True,
    tol=0.001, verbose=False)

# Support Vector Classfier - BUT with probability = True so a soft voting scheme can be used. 
svc_clf_proba = SVC(C=100, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
    max_iter=-1, probability=True, random_state=42, shrinking=True,
    tol=0.001, verbose=False)

# Decision Tree - Submission 2
tree_clf = DecisionTreeClassifier(
    criterion = 'gini', max_depth = None, min_samples_leaf = 5, min_samples_split = 2,random_state=42)

# Voting Ensemble Classfier - Hard Voting 
voting_clf = VotingClassifier(
    estimators = [('sub1',svc_clf), ('sub2',tree_clf)],
    voting='hard')

# Voting Ensemble Classifier - Soft Voting
voting_clf_soft = VotingClassifier(
    estimators = [('sub1_proba',svc_clf_proba), ('sub2',tree_clf)],
    voting='soft')

# Fit both voting ensemble methods

voting_clf.fit(train_features_prep,target_features)
voting_clf_soft.fit(train_features_prep,target_features)



# Loop through classifer performing 5 fold Cross Validation 
for clf in (svc_clf,tree_clf,voting_clf,voting_clf_soft):
    cvs = cross_val_score(clf, train_features_prep,target_features,cv =5,scoring='accuracy')
    print(clf.__class__.__name__,cvs, '\n' ,'Mean:' , cvs.mean())
print('\n')




# ------------------------- Learning Curves -------------------------------
fig,ax = plt.subplots()
ax.set_title('Submission 3 - Hard Voting Classfier')
ax.set_ylim(0.6,1.02)
ax.set_xlim(0,720)
mlib.plot_learning_curve(
    voting_clf, train_features_prep, target_features, 'accuracy', 5, ax) 

# Learning Curves 
fig,ax = plt.subplots()
ax.set_title('Submission 3 - Soft Voting Classfier')
ax.set_ylim(0.6,1.02)
ax.set_xlim(0,720)
mlib.plot_learning_curve(
    voting_clf_soft, train_features_prep, target_features, 'accuracy', 5, ax) 



# --------------- Predicting on test data and save submission -------------------------

predictions_hard = voting_clf.predict(test_features_prep)
predictions_soft = voting_clf_soft.predict(test_features_prep)
mlib.Pred_to_Kaggle_Format(predictions_hard,'Submissions/Submission_3.csv')
mlib.Pred_to_Kaggle_Format(predictions_soft,'Submissions/Submission_4.csv')


#%% 

# ------------------------ Finish --------------------------------------------
    
print('\n') 
print('Script runtime:', (time.time()-start)/60)
print('\n' '\n')
print('----------------- End --------------------')


















