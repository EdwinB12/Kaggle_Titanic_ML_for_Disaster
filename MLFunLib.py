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
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

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

def Full_PipeLine(df, feature_list, target_list,  num_pipe, cat_pipe):
    '''
    Takes data in df form, splits into numerical or categorical feature
    depending on dtype, combines given individual pipelines for num and cat data into a full 
    pipeline. Outputs full pipeline, both training and target features (without the column headers),
    and also a list of training feature columns. 

    Parameters
    ----------
    df : dataframe
        dataframe containing data desired to be transformed by pipeline.
    feature_list : String List
        List of dataframe columns names corresponding to features.
    target_list : String list
        List of dataframe columns names corresponding to the target features..
    num_pipe : PipeLine output
        PipeLine for numnerical data.
    cat_pipe : PipeLine output
        Pipeline for categorical data.

    Returns
    -------
    full_pipeline : Pipeline
        Pipeline for fitting and transforming data.
    train_features : Numpy array
        Array of training features (no column headers).
    target_features : Numpy array
        Array of target feature (no column headers)..
    post_trans_feature_list : List
        List of training features column headers.

    '''
    
    
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
    
    # Creating list of output features in the order they will appear after transformation 
    post_trans_feature_list = np.concatenate([numeric_features,categoric_features])
    
    return full_pipeline, train_features, target_features, post_trans_feature_list

#--------------------------------------------------------------------------------------------------
    
# -------------------------------- Saved CSV for Kaggle Submission --------------------------------

# Function preparing predictons for Kaggle Submission 
def Pred_to_Kaggle_Format(predictions,csv_path):
    '''
    Function to write predictions for the Kaggle Titanic dataset in correct csv format. 
    
    Parameters
    ----------
    predictions : 1D array
        Array of predicitons with no column headers. Should be 1s or 0s only. 
    csv_path : string
        Destination and name of csv file to output.

    '''
    
    # Take passanger ID from original test dataset and flatten
    df = pd.read_csv('Original_Data/test.csv')
    pass_id = df['PassengerId'].to_numpy().ravel()
    
    # Create new dataframe with passengerID and Survived Columns
    pred_df = pd.DataFrame({'PassengerId' : pass_id, 'Survived' : predictions})
    pred_df.to_csv(csv_path,index=False)

#---------------------------------------------------------------------------------------------------
    
# ------------------------------ Scree plot --------------------------------------------------------

def Scree_Plot(data,n_components=None):
    '''
    Plots scree plot of principal components and their respective variance ratios. 

    Parameters
    ----------
    data : Array
        Array of training data after prep - No columns labels.
    n_components : int or float, optional
        DESCRIPTION. The default is None. Check PCA n_components documentation. 

    '''
    #Calculate PCA for user defined number of components
    pca = PCA(n_components=n_components) 
    pca.fit(data)

    # setting variance ratios
    y =  pca.explained_variance_ratio_
    
    # Creating string list of x-axis
    x = list()
    for i in range(1,(len(y)+1)):
        x.append('PC' + str(i))
        
    # Plotting
    fig,ax = plt.subplots()
    ax.bar(x,y)
    ax.set_title('Scree Plot')
    ax.set_xlabel('Principal Components')
    ax.set_ylabel('Variance Ratio')
    ax.spines['right'].set_visible(False) # Removing right and top spines
    ax.spines['top'].set_visible(False)
    
#---------------------------------------------------------------------------------------------------
    
#--------------------- Plot Principal Components Scatter Matrix ------------------------------------    
    
    
def PC_CrossPlotting_Color(train_data,target_data,n_components=None): 
    '''
    Produces scatter matrix of principal components derived from the training data. 
    The datapoints will be coloured by their respective target feature value. 

    Parameters
    ----------
    train_data : Array
        Train data after prep. No Dataframe or column labels. 
    target_data : 1D Array
        1D array of target feature values.
    n_components : TYPE, optional
        DESCRIPTION. The default is None. Check PCA n_components documentation.

    '''
    
    # Performing PCA on user defined train_data with user stated n_components
    pca=PCA(n_components=n_components)
    pca_values =  pca.fit_transform(train_data)
    
    # Adding user defined target values to be used as 'hue'
    hue_values = np.reshape(target_data,(len(target_data),1))
    df_values = np.append(pca_values ,hue_values,axis=1)
    
    # Creating list of PC labels
    x = list()
    for i in range(1,(df_values.shape[1])):
        x.append('PC' + str(i))
    
    #Adding the hue variable pandas column names at the end    
    x.append('hue_var')
    
    # Creating df with the data and columns
    df = pd.DataFrame(data =df_values, columns = x )
    
    # Plotting
    sns.set(style="ticks")
    sns.pairplot(df, hue='hue_var')
   
#----------------------------------------------------------------------------------------------------
# --------------- Function to perform Feature Engineering on a dataframe ----------------------------
    
def Feature_Engineering(df): 
    '''
    Function to conveniently run feature engineering. 

    Parameters
    ----------
    df : Pandas Dataframe
        Input Pandas Dataframe directly from load.

    Returns
    -------
    df : Pandas Dataframe
        Data with feature engineering applied. This dataframe can now go into pipeline. 

    '''
    
    # Drop Passenger ID and Cabin 
    df = df.drop(['PassengerId','Ticket','Cabin'],axis=1)
    
    # ---------------- Create 'Title' feature and remove 'Name' ------------------
    
    # Extract Title from Name
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    
    # Group any title that isn't Mr, MRs, Miss or Master in group 'other'. 1st correct some typos etc. 
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    df['Title'] = df['Title'].replace(['Don', 'Rev', 'Dr', 'Mme', 'Ms',
           'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'Countess',
           'Jonkheer'],'Other')
    
    # Remove Name Column
    df = df.drop('Name',axis=1)
    
    # ----------------- Sorting Age into Brackets ----------------
    
    # Bin age into discrete values and look at mean survival
    df['AgeBand'] = pd.cut(df['Age'], 5)
    
    # Define New Column AgeInt and assign values based on the Ageband analysis above. Remove Ageband.
    df['AgeInt'] = df['Age']
       
    df.loc[ df['Age'] <= 16, 'AgeInt'] = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'AgeInt'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'AgeInt'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'AgeInt'] = 3
    df.loc[df['Age'] > 64, 'AgeInt'] = 4
    df = df.drop('AgeBand',axis=1)
    
    # ---------------- IsAlone Feature - 1 if alone, 0 if not -------------------------
    
    # Assign all rows 0 as IsAlone value. 
    df['IsAlone'] = 0
    
    # Make IsAlone = 1 if SibSp and Parch both = 0
    df.loc[(df['SibSp'] == 0) & (df['Parch'] <= 0), 'IsAlone'] = 1
        
    # ----------------------------- Fare Bands ----------------------------
    
    #df['FareBand'] = pd.cut(df['Fare'],50)
    #df[['FareBand','Survived']].groupby['FareBand'].mean()
    df.loc[ df['Fare'] <= 10, 'FareInt'] = 0
    df.loc[(df['Fare'] > 10) & (df['Fare'] <= 30), 'FareInt'] = 1
    df.loc[(df['Fare'] > 30) , 'FareInt'] = 2

    return df


#------------------------------------------------------------------------------------------
# Function to plot mean survived value for each value of the variable (discrete data only)


def Survive_av_plot(df,var): 
    
    count = 0
    var_values = df[var].unique()
    Sur_mean = np.empty([len(var_values)])
    for i in var_values:
        Sur_mean[count] = (df[df[var] == i].Survived).mean()
        count=count+1
    
    plt.scatter(x=var_values,y = Sur_mean)
    plt.ylim(0,1)
    plt.title(var)

