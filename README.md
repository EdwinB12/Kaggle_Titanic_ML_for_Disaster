**---------------------------------- IN PROGRESS -----------------------------------------**

# Kaggle: Titanic ML for Disaster

The goal of this project is to predict who would survive onboard the famous Titanic disaster. This project is a popular Kaggle competition called 'Titanic: Machine Learning from Disaster'. https://www.kaggle.com/c/titanic 

I used this project to practice my data handling skills in Python, aswell as gaining experience using the Scikit-Learn for a binary classification problem. 



## Introduction to the challenge

The aim of this project is to use machine learning techniques to predict whether a person would have survived or not survived the titanic disaster. 1502 out of 2224 passengers and crew onboard the Titanic died. A training dataset is provided including whether each person survived. A Test dataset is also provided but without any indication of survival. The final score is determined by what percentage are correctly predicted to have survived. 

**NOTE: A Survived value of 1 means the person survived and a 0 means they didn't survive.** 

### Data

Two datasets are provided: 
- Training dataset: 891 passengers information provided
- Test dataset: 418 passengers information provided

Details - https://www.kaggle.com/c/titanic/data 

## My Approach 

I split the project into two phases: 
- Data Preparation
  - Feature Engineering 
  - Data Cleaning and Formatting
  
- Training and Evaluation 
  - Searching for optimal training parameters for chosen method
  - Evaluation by 5 or 10 fold cross validation 
  - Assess learning curves of optimal learning parameters

### Data Prep 

I prepped the data in two different ways. 
- Prep 1 : Submission 1-5
- Prep 2 : Submissions 6-10

#### Prep 1
*Feature Engineering*

This was a very simple approach to preparing the data, designed to create a benchmark value.  
The feature engineering simply involved either accepting or rejecting a provided feature. Simple linear relationships were assessed between each feature before accepting or rejecting. 

6 Features were accepted: 
- Pclass
- Sex
- SibSp
- Parch
- Fare
- Embarked

All other feature were thrown away. **No new features were created or calculated**

*Data Cleaning and Formatting*

Any missing values (there isn't many for these features), was imputed by using the most common value for categorical data and the mean for numerical data.
Standardize Scaling was applied where appropriate, although decision trees were most used for training and therefore scaling was not required. 
Ordinal Encoding was often adequate for this data although OneHotEncoding is easily applied if thought to be more suitable. 


#### Prep 2 
*Feature Engineering*

More effort was put into this version of prepping the data. Particular effort was put into adapting the Age, Fare, Name, SibSp and Parch features to extract more useful information. 

Name: The actual name of the passengers is probabaly unimportant but the title can indicate both sex and potential age. Titles were extracted from the names. The categories were split into Mr, Master, Mrs, Miss and other. 

Fare: Opposed to Prep 1, I took the raw fare values and split them into 3 bands. This somewhat acted as a similar feature to Pclass. The mean survival was 0.58 for people paying over £30 for their ticket opposed to 0.2 when payed less than £10. 

Age: Previously unused in Prep 1 due to about 10% missing values and little correlation with whether people survived. Prep 2 divided age into 5 bands after noticing people aged over 64 had a mean survival value of 0.09 opposed to under 16s mean survival value of 0.55. 

IsAlone: Weak correlations between the SibSp and Parch features and survival observed. Therefore, these features were used to create an 'IsAlone' feature which indicates whether somebody is travelling alone or not. Mean survival values of 0.5 and 0.3 respectively for people travelling alone and not alone. 

7 features used were: Pclass, Sex, AgeInt (Age split into bands), IsAlone, FareInt (Fare split into bands), Emabarked and Title. 

*Data Cleaning and Formatting*

Any missing values were imputed using the most common value. Approximately 10% of the Age data was missing and therefore where Age data was missing, the instance was placed in the most populous band. A more sophiscated approach may use the title to guess at the age but I didn't bother doing it here. 
Standardize Scaling was applied where appropriate, although decision trees were most used for training and therefore scaling was not required. 
Ordinal Encoding was often adequate for this data although OneHotEncoding is easily applied if thought to be more suitable. 

### Training and Evaluation

Training generally consisted of a grid search for optimal learning parameters for the chosen algorithm using 5 fold cross validation. 
A number of learning algorithms were tested included a number of ensemble and boosting methods. 
The trained models were assessed by looking at cross validation accuracy scores aswell as training curves. 
*Further validation work is required, including an assessment of precision, recall and ROC curves.* 

The explored algorithms were: 
- Logistic regression with SGD
- SVM Classifier
- Decision Tree
- Soft and Hard Voting Classifiers
- Random Forest
- Ada Boosting
- Gradient Boosting 

## Submissions and Scores

10 Submissions were made in total to date with a high score of 0.789. Final Parameters can be found on the Kaggle submissions tab.
The following table summarises my current submissions: 
- Column 1 : Submission Number on Kaggle 
- Column 2 : Data prep as defined above, either method 1 or 2. 
- Column 3 : Algorithm used, exact set up can be found in corresponding submission script
- Column 4 : 5 fold cross validation accuracy score on the training data. 
- Column 5 : Kaggle score given by predicting results on the unseen testing data. 

Learning curves for each submission can be found in the Figures Folder. 

|   | Data Prep  | Algorithm  | Cross Val Score  | Kaggle Score  |
|:-:|---|---|---|---|
| [Sub1](Submission_1.py)  | Prep 1 | SVM  | 0.788 | 0.75119 |
| [Sub2](Submission_2.py)  | Prep 1 | Decision Tree  | 0.811 | 0.76555  |
| [Sub3](Submission_3_4.py)  | Prep 1 | 'Hard' Voting Ensemble Model - Sub 1 & 2   |  0.801 | 0.75119  |
| [Sub4](Submission_3_4.py)  | Prep 1 | 'Soft' Voting Ensemble Model - Sub 1 & 2  | 0.800  | 0.75598 |
| [Sub5](Submission_5.py)  | Prep 1 | Random Forest |0.808 | 0.77511  |
| [Sub6](Submission_6_7_8.py)  | Prep 2 | SVM | 0.811  | 0.78468  |
| [Sub7](Submission_6_7_8.py)  | Prep 2 | Decision Tree | 0.822  | 0.77511   |
| [Sub8](Submission_6_7_8.py)  | Prep 2 | Random Forest  | 0.818  | 0.78947  |
| [Sub9](Submission_9_10.py)  | Prep 2 | Ada Boost - Decision Tree  | 0.816  | 0.72727  |
| [Sub10](Submission_9_10.py) | Prep 2 | Gradient Boost - Decision Tree  | 0.828  | 0.78947  |



## Repository Layout

The scripts at base level named 'Submission 1', 'Submission 2', etc. are the scripts run the generate my Kaggle submissions. 

## Author

Edwin Brown - Previously a geophysicist but looking to transition into a more focused data science career using machine learning workflows to solve challenging problems. 

## License

This project is licensed under the MIT License. See the [LICENSE.md](LICENSE.md) file for details. 

# Acknowledgments

Kaggle for setting the challenge and providing the data in an easy to use format. 
