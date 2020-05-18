# -*- coding: utf-8 -*-
"""
Created on Sun May 17 14:14:40 2020

Data Vis Testing 

@author: edwin
"""

#%% Experimenting with cross-plotting data

# Collecting both prepped train data and labels into a dataframe
import pandas as pd
import seaborn as sns
data = train_features_prep
columns = post_transform_train_features
train_features_prep_df= pd.DataFrame(data=data, columns=columns)
train_features_prep_df['Survived'] = target_features



sns.catplot(
    x= 'Sex', hue = 'Embarked', col = 'Survived', data = train_features_prep_df, kind='count')



sns.catplot(x="Sex", y="Survived", col="Pclass",
                data=train_features_prep_df, saturation=.5,
                kind="bar", ci=None, aspect=.6)