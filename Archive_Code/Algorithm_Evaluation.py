# -*- coding: utf-8 -*-
"""
Created on Tue May 12 10:32:40 2020

Kaggle Titanic: Machine Learning for a Disaster
More information is here: https://www.kaggle.com/c/titanic/data

Document Objective: Initial testing of different model types and optimisers
                    by plotting learning curves. 

@author: edwin
"""

#%% ------------- Packages and start of script ----------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import time
import pickle
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC,LinearSVC 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

print('\n')
print('--------------- Start --------------------')
print('\n' '\n')
start=time.time()

# Loading in S2 dataframe - no feature scaling
with open('Pickled_Files/train_df_S2.txt', 'rb') as myFile:
    train_df_S2 = pickle.load(myFile)


#%% Functions 

# Function for plotting training and validation curves
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

def feature6_plot(algorithm,title):
    fig,([ax1,ax2],[ax3,ax4],[ax5,ax6]) = plt.subplots(3,2,figsize = (15,10))
    ax1.set_ylim(0.6,1.02)
    ax2.set_ylim(0.6,1.02)
    ax3.set_ylim(0.6,1.02)
    ax4.set_ylim(0.6,1.02)
    ax5.set_ylim(0.6,1.02)
    ax6.set_ylim(0.6,1.02)
    plot_learning_curve(algorithm, training_features_SS_1, target_feature,'accuracy',10, ax1)
    plot_learning_curve(algorithm, training_features_SS_2, target_feature,'accuracy',10, ax2)
    plot_learning_curve(algorithm, training_features_SS_3, target_feature,'accuracy',10, ax3)
    plot_learning_curve(algorithm, training_features_SS_4, target_feature,'accuracy',10, ax4)
    plot_learning_curve(algorithm, training_features_SS_5, target_feature,'accuracy',10, ax5)
    plot_learning_curve(algorithm, training_features_SS_6, target_feature,'accuracy',10, ax6)
    ax1.set_title("1 Feature")
    ax2.set_title("2 Feature")
    ax3.set_title("3 Feature")
    ax4.set_title("4 Feature")
    ax5.set_title("5 Feature")
    ax6.set_title("6 Feature")
    fig.suptitle(title,fontsize = 'xx-large' )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


#%% 
    
#--------------- Organising training data ------------------------

# Assigning training and target features to seperate numpy arrays
target_feature = train_df_S2['Survived'].to_numpy()
training_features = train_df_S2[['Sex','Pclass','Fare','Embarked','Parch','SibSp']].to_numpy()

# Feature scale training features
scalar = StandardScaler()
training_features_SS = scalar.fit_transform(training_features)

# Create different training sets with increasing number of features, starting with the highest R valued features
training_features_SS_1 = training_features_SS[:,:1] # Sex
training_features_SS_2 = training_features_SS[:,:2] # Sex and Pclass
training_features_SS_3 = training_features_SS[:,:3] # Sex and Pclass and Fare
training_features_SS_4 = training_features_SS[:,:4] # Sex and Pclass, Fare and Embarked
training_features_SS_5 = training_features_SS[:,:5] # Sex and Pclass, Fare, Embarked and Parch
training_features_SS_6 = training_features_SS[:,:6] # Sex and Pclass, Fare, Embarked, Parch and SibSp


# ---------------- Parametirising ML Algorithms -------------------

# Perform linear classifier with SGD and hinge loss (equivalent to linear SVM)
sgd_clf = SGDClassifier(loss = 'hinge',random_state = 42)

#Perform linear classifier with SGD and log loss (logistic regression)
sgd_clf_log = SGDClassifier(loss = 'log',random_state = 42)

# Perform Logistic Regression
log_reg_liblin = LogisticRegression(random_state = 42, solver = 'liblinear')
log_reg_newton = LogisticRegression(random_state = 42, solver = 'newton-cg')

#Perform RandomForestClassifier 
forest_clf = RandomForestClassifier(random_state=42)

#Perform SVM
svm_clf = SVC()
svm_clf_rbf = SVC(kernel='rbf',C=1)
svm_clf_rbf_C100 = SVC(kernel='rbf',C=100)
svm_clf_lin = SVC(kernel='linear',C=1)
svm_clf_poly = SVC(kernel='poly',C=1,degree=3)

#Perform LinearSVM
lin_svm_clf = LinearSVC()
lin_svm_clf_hinge = LinearSVC(loss = 'hinge')


#%% 

# -------------- Comparing learning curves from different algorithms on S2 Data -----------------
fig,([ax1,ax2],[ax3,ax4]) = plt.subplots(2,2,figsize = (24,12))
ax1.set_ylim(0.4,1.1)
ax2.set_ylim(0.4,1.1)
ax3.set_ylim(0.4,1.1)
ax4.set_ylim(0.4,1.1)
plot_learning_curve(sgd_clf, training_features, target_feature,'accuracy',10, ax1)
plot_learning_curve(sgd_clf_log, training_features, target_feature,'accuracy',10, ax2)
plot_learning_curve(forest_clf, training_features, target_feature,'accuracy',10, ax3)
plot_learning_curve(svm_clf, training_features, target_feature,'accuracy',10, ax4)
ax1.set_title("Learning Curve - SGD Classifier: hinge loss")
ax2.set_title("Learning Curve - SGD Classifier: log loss")
ax3.set_title("Learning Curve - Random Forest")
ax4.set_title("Learning Curve - SVM")


#%% -------------- Comparing learning curves from different algorithms on S2 Data with SS -----------------

fig,([ax1,ax2],[ax3,ax4]) = plt.subplots(2,2,figsize = (24,12))
ax1.set_ylim(0.4,1.1)
ax2.set_ylim(0.4,1.1)
ax3.set_ylim(0.4,1.1)
ax4.set_ylim(0.4,1.1)
plot_learning_curve(sgd_clf, training_features_SS, target_feature,'accuracy',10, ax1)
plot_learning_curve(sgd_clf_log, training_features_SS, target_feature,'accuracy',10, ax2)
plot_learning_curve(forest_clf, training_features_SS, target_feature,'accuracy',10, ax3)
plot_learning_curve(svm_clf, training_features_SS, target_feature,'accuracy',10, ax4)
ax1.set_title("Learning Curve - SGD Classifier: hinge loss")
ax2.set_title("Learning Curve - SGD Classifier: log loss")
ax3.set_title("Learning Curve - Random Forest")
ax4.set_title("Learning Curve - SVM")


#%% -------------- Comparing learning curves from different algorithms on S2 Data with SS -----------------
# Linear SVM replaces Random Forest

fig,([ax1,ax2],[ax3,ax4]) = plt.subplots(2,2,figsize = (24,12))
ax1.set_ylim(0.4,1.1)
ax2.set_ylim(0.4,1.1)
ax3.set_ylim(0.4,1.1)
ax4.set_ylim(0.4,1.1)
plot_learning_curve(sgd_clf, training_features_SS, target_feature,'accuracy',10, ax1)
plot_learning_curve(lin_svm_clf_hinge, training_features_SS, target_feature,'accuracy',10, ax2)
plot_learning_curve(lin_svm_clf, training_features_SS, target_feature,'accuracy',10, ax3)
plot_learning_curve(svm_clf, training_features_SS, target_feature,'accuracy',10, ax4)
ax1.set_title("Learning Curve - SGD Classifier: hinge loss")
ax2.set_title("Learning Curve - Linear SVM: hinge loss")
ax3.set_title("Learning Curve - Linear SVM")
ax4.set_title("Learning Curve - SVM")


#%% -------------- Comparing learning curves from different SVM kernals ------------------

fig,([ax1,ax2],[ax3,ax4]) = plt.subplots(2,2,figsize = (24,12))
ax1.set_ylim(0.6,1.1)
ax2.set_ylim(0.6,1.1)
ax3.set_ylim(0.6,1.1)
ax4.set_ylim(0.6,1.1)
plot_learning_curve(lin_svm_clf_hinge, training_features_SS, target_feature,'accuracy',10, ax1)
plot_learning_curve(svm_clf_rbf, training_features_SS, target_feature,'accuracy',10, ax2)
plot_learning_curve(svm_clf_lin, training_features_SS, target_feature,'accuracy',10, ax3)
plot_learning_curve(svm_clf_poly, training_features_SS, target_feature,'accuracy',10, ax4)
ax1.set_title("Learning Curve - SVM")
ax2.set_title("Learning Curve - SVM RBF")
ax3.set_title("Learning Curve - SVM Linear")
ax4.set_title("Learning Curve - SVM poly")

#%% ------------- Comparing learning curves with different number of features using a svm_clf_rbf ----------------

feature6_plot(sgd_clf,'sgd_clf')
feature6_plot(svm_clf_rbf,'svm_clf_rbf' )
feature6_plot(svm_clf_lin, 'svm_clf_lin')
feature6_plot(svm_clf_poly, 'svm_clf_poly')
feature6_plot(forest_clf, 'forest_clf' )
feature6_plot(lin_svm_clf_hinge, 'lin_svm_clf_hinge')

#%% Comparing logistic regression with different solvers

fig,([ax1,ax2],[ax3,ax4]) = plt.subplots(2,2,figsize = (24,12))
ax1.set_ylim(0.6,1.1)
ax2.set_ylim(0.6,1.1)
ax3.set_ylim(0.6,1.1)
ax4.set_ylim(0.6,1.1)
plot_learning_curve(log_reg_liblin, training_features_SS, target_feature,'accuracy',10, ax1)
plot_learning_curve(log_reg_newton, training_features_SS, target_feature,'accuracy',10, ax2)
plot_learning_curve(sgd_clf_log, training_features_SS, target_feature,'accuracy',10, ax3)
plot_learning_curve(lin_svm_clf_hinge, training_features_SS, target_feature,'accuracy',10, ax4)
ax1.set_title("Log Regression - LibLin")
ax2.set_title("Log Regression - Newton")
ax3.set_title("Log Regression - SGD")
ax4.set_title("lin_svm_clf_hinge")

#%% 

fig,([ax1,ax2]) = plt.subplots(1,2,figsize = (24,12))
ax1.set_ylim(0.6,1.1)
ax2.set_ylim(0.6,1.1)
plot_learning_curve(svm_clf_rbf, training_features_SS, target_feature,'accuracy',10, ax1)
plot_learning_curve(svm_clf_rbf_C100, training_features_SS, target_feature,'accuracy',10, ax2)
ax1.set_title("C1")
ax2.set_title("C100")

#%% Finish
    
print('\n') 
print('Script runtime:', (time.time()-start)/60)
print('\n' '\n')
print('----------------- End --------------------')