# -*- coding: utf-8 -*-
"""
DS 861 - Prof Minh Pham - Spring 2023
Project - Binary Classification - Adult Census
Name: Saksham Motwani
"""
##Importing libraries
 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import math
import itertools
from sklearn import metrics
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from scipy.stats import randint

# Entering data into a dataframe
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

train_data = pd.read_csv('adult.data', sep=",\s", header = None, names = column_names, engine = 'python')
test_data = pd.read_csv('adult.test', sep=",\s", header = None, names = column_names, engine = 'python')
test_data['income'].replace(regex = True, inplace = True, to_replace = r'\.', value = r'') # The income attribute in the test data ends with a period sign. Replacing period sign with empty string.

df = pd.concat([train_data,test_data])
df.reset_index(inplace = True, drop = True)

# %%

# PRELIMINARY DATA ANALYSIS

print(df) # printing dataframe
print(df.shape) # printing dimesnions of data. 48842 rows and 15 columns

# Columns and their types
# age, fnlwgt, education-num, capital-gain, capital-loss, and hours-per-week are integer columns. There are no float datatype.
# workclass, education, marital-status, occupation, relationship, race, sex, native-country and income are of object datatype i.e. categorical
print(df.info()) 

print(df.isnull().sum()) #Checking missing values in dataframe. There are no missing values

print('Unique values in every column : \n') ##printing unique values in columns
for col in df:
    print(col,':' , df[col].unique())

df_copy = df.copy() # creating copy of original master data

# %%

# HANDLING MISSING AND DUPLICATE VALUES

# Though the dataset does not have any null/missing values, upon looking closer, there are a lot of '?' values.
# We will replace these '?' values with np.nan so we can drop them.
df_copy.replace('?', np.nan, inplace = True)

print(df_copy.isna().sum()) #2799 missing values in column workclass, 2809 in column occupation, adn 857 in column native-country.

df_copy = df_copy.dropna() # Dropping missing values
print(df_copy.shape) # printing dimesnions of data. 45222 rows and 15 columns (after dropping missing values)

# Dropping duplicate rows (keep first) and resetting index
df_copy.drop_duplicates(keep = 'first', inplace = True)
df_copy.reset_index(drop = True, inplace = True)

print(df_copy.shape) # printing dimesnions of data. 45175 rows and 15 columns (after dropping duplicates)

# %%

# FEATURE ENGINEERING

# Column 'education'
# Looking at the education-num column, which was present in the dataset, and gives a hierarchial number to every category of education, we re-categorize column 'education' into 
# 'School-dropout'   (levels 1-8 of education-num)
# 'High-school-grad' (level 9 of education-num)
# 'Associate-degree' (level 10,11,12 of education-num)
#  We leave categories 'Bachelors', 'Masters', 'Prof-school', and 'Doctorate' of column 'education' which correspond to levels 13, 14, 15, 16 of education-num as they are.
df_copy['education'] = df_copy['education'].replace(['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th','10th', '11th', '12th'], 'School-dropout')
df_copy['education'] = df_copy['education'].replace('HS-grad', 'High-school-grad')
df_copy['education'] = df_copy['education'].replace(['Assoc-voc', 'Assoc-acdm', 'Some-college'], 'Associate-degree')

# Column 'education-num'
# This column can now be dropped since the information here is already captured in the column 'education'
df_copy.drop('education-num', axis = 1, inplace = True)

# Column 'marital-status'
# We re-categorize 'marital-status' into 
# 'Married' if status is 'Married-civ-spouse', 'Married-AF-spouse', 'Married-spouse-absent', or 'Separated'
# 'Single' if status is 'Never-married','Divorced',or 'Widowed'
df_copy['marital-status']= df_copy['marital-status'].replace(['Married-civ-spouse', 'Married-AF-spouse','Married-spouse-absent', 'Separated'], 'Married')
df_copy['marital-status']= df_copy['marital-status'].replace(['Never-married', 'Divorced', 'Widowed'], 'Single')

# Column 'native-country'
# We re-categorize 'native-country' into
# 'US' if 'United-States'
# 'Others' if any other value
df_copy['native-country'] = df_copy['native-country'].replace('United-States','US')
df_copy['native-country'] = df_copy['native-country'].replace(['Philippines', 'Germany', 'Puerto-Rico', 'Canada', 'El-Salvador',
'India', 'Cuba', 'England', 'China', 'South', 'Jamaica', 'Italy', 'Dominican-Republic', 'Japan', 'Guatemala', 'Poland',
'Vietnam', 'Columbia', 'Haiti', 'Portugal', 'Taiwan', 'Iran', 'Greece', 'Nicaragua', 'Peru', 'Ecuador', 'France', 'Ireland',
'Hong', 'Thailand', 'Cambodia', 'Trinadad&Tobago', 'Laos', 'Yugoslavia', 'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Honduras',
'Hungary', 'Holand-Netherlands','Mexico'],'Other')

# Column 'workclass'
# We re-categorize 'workclass' into
# 'Private' if workclass is 'Private' (no change here)
# 'Without-pay' if workclass is 'Without-pay' (no change here; very very less records here)
# 'Government' if workclass is 'Local-gov','State-gov','Federal-gov'
# 'Self-employed' if workclass is 'Self-emp-not-inc' or 'Self-emp-inc'
df_copy['workclass'] = df_copy['workclass'].replace(['Local-gov','State-gov','Federal-gov'],'Government')
df_copy['workclass'] = df_copy['workclass'].replace(['Self-emp-not-inc','Self-emp-inc'],'Self-employed')

# %%

# SUMMARY STATISTICS AND CORRELATION FOR NUMERIC VARIABLES
distribution = df_copy.describe()
skew = df_copy.skew().to_frame().T.rename(index={0: 'skew'})
kurtosis = df_copy.kurtosis().to_frame().T.rename(index={0: 'kurtosis'})
print(pd.concat([distribution, skew, kurtosis]).T)

# We see that none of the numerical columns are highly correlated which is a good thing because we can have a more stable model later
print(df_copy.corr())

# %%

# Exploratory Data Analysis

# Distribution of categorical variables
fields = df_copy.select_dtypes(exclude="number").columns
plt.subplots(nrows = 3, ncols = 3, figsize = (16,14))
for i in range(1,len(fields)+1):
    plt.subplot(3,3,i)
    df_copy[fields[i-1]].value_counts().sort_values().plot.bar(color='orange')  
    plt.xticks(rotation=90)
    plt.title(fields[i-1])
plt.tight_layout()
plt.show()

# Distribution of numerical features of the dataset. capital-gain, capital-loss, fnlwgt is right skewed
distribution = df_copy.hist(edgecolor = 'black', linewidth = 1.2, color = 'c')
fig = plt.gcf()
fig.set_size_inches(12,12)
plt.show()

# Distribution of response variable (income)
# We see that the response variable is imbalanced (25% and 75%)
plt.pie(df_copy['income'].value_counts(), labels = df_copy['income'].unique(), autopct='%1.1f%%')
plt.legend(df_copy['income'].unique())
plt.title('Distribution of Data in Income')
plt.show()

# Education v/s Income
# This graph shows the proportion of income classes accross education levels. As expected, as the education level increases,
# the proportion of people who earn more than 50k a year also increases. But what is interesting is that only after a masters degree
# the proportion of people earning more than 50k a year is a majority.
education = round(pd.crosstab(df_copy['education'], df_copy['income']).div(pd.crosstab(df_copy['education'], df_copy['income']).apply(sum,1),0),2)
education.sort_values(by = '>50K', inplace = True)
ax = education.plot(kind ='bar', title = 'Proportion distribution across education levels', figsize = (10,8))
ax.set_xlabel('Education level')
ax.set_ylabel('Proportion of population')

# Gender v/s Income
# This graph shows the proportion of income classes accross genders. We can see there exists a wide gap
# between males and females (Proportion of males earning more than 50k is more than double of females)
gender = round(pd.crosstab(df_copy['sex'], df_copy['income']).div(pd.crosstab(df_copy['sex'], df_copy['income']).apply(sum,1),0),2)
gender.sort_values(by = '>50K', inplace = True)
ax = gender.plot(kind ='bar', title = 'Proportion distribution across gender levels')
ax.set_xlabel('Gender level')
ax.set_ylabel('Proportion of population')

# Workclass v/s Income
workclass = round(pd.crosstab(df_copy['workclass'], df_copy['income']).div(pd.crosstab(df_copy['workclass'], df_copy['income']).apply(sum,1),0),2)
workclass.sort_values(by = '>50K', inplace = True)
ax = workclass.plot(kind ='bar', title = 'Proportion distribution across workclass levels', figsize = (10,8))
ax.set_xlabel('Workclass level')
ax.set_ylabel('Proportion of population')

# Occupation v/s Income
# This graph shows the proportion of income classes accross occupations.
# We can see that for occupations such as 'Prof-specialty' and 'Exec-managerial', the proportion of people whose salary is higher than 50k per year is more than other occupations.
occupation = round(pd.crosstab(df_copy['occupation'], df_copy['income']).div(pd.crosstab(df_copy['occupation'], df_copy['income']).apply(sum,1),0),2)
occupation.sort_values(by = '>50K', inplace = True)
ax = occupation.plot(kind ='bar', title = 'Proportion distribution across Occupation levels', figsize = (10,8))
ax.set_xlabel('Occupation level')
ax.set_ylabel('Proportion of population')

# Marriage v/s Income
# This graph shows the proportion of income classes accross marital-status. We can see the proportion of married 
# people having income more than 50K is higher than single people having income more than 50K
marriage = round(pd.crosstab(df_copy['marital-status'], df_copy['income']).div(pd.crosstab(df_copy['marital-status'], df_copy['income']).apply(sum,1),0),2)
marriage.sort_values(by = '>50K', inplace = True)
ax = marriage.plot(kind ='bar', title = 'Proportion distribution across Marital Status', figsize = (10,8))
ax.set_xlabel('Marital Status')
ax.set_ylabel('Proportion of population')

# Hours-per-week v/s Income
hours_per_week = round(pd.crosstab(df_copy['hours-per-week'], df_copy['income']).div(pd.crosstab(df_copy['hours-per-week'], df_copy['income']).apply(sum,1),0),2)
ax = hours_per_week.plot(kind ='bar', title = 'Proportion distribution across Hours per week', figsize = (20,12))
ax.set_xlabel('Hours per week')
ax.set_ylabel('Proportion of population')

# %%

# PREPARING THE DATA FOR MODELING

# Creating Dummy Variables (also for the response variable)

# Reference value for workclass is Government
workclass_dummy = pd.get_dummies(df_copy['workclass'], prefix = 'workclass')
del workclass_dummy['workclass_Government']

# Reference value for education is School-dropout
education_dummy = pd.get_dummies(df_copy['education'], prefix = 'education')
del education_dummy['education_School-dropout']

# Reference value for marital-status is Single
maritalstatus_dummy = pd.get_dummies(df_copy['marital-status'], prefix = 'marital-status')
del maritalstatus_dummy['marital-status_Single']

# Reference value for occupation is Craft-repair
occupation_dummy = pd.get_dummies(df_copy['occupation'], prefix = 'occupation')
del occupation_dummy['occupation_Craft-repair']

# Reference value for relationship is Unmarried
relationship_dummy = pd.get_dummies(df_copy['relationship'], prefix = 'relationship')
del relationship_dummy['relationship_Unmarried']

# Reference value for race is Other
race_dummy = pd.get_dummies(df_copy['race'], prefix = 'race')
del race_dummy['race_Other']

# Reference value for sex is Female
sex_dummy = pd.get_dummies(df_copy['sex'], prefix = 'sex')
del sex_dummy['sex_Female']

# Reference value for native-country is Other
nativecountry_dummy = pd.get_dummies(df_copy['native-country'], prefix = 'native-country')
del nativecountry_dummy['native-country_Other']

# Reference value for income is <=50K
income_dummy = pd.get_dummies(df_copy['income'], prefix = 'income')
del income_dummy['income_<=50K']

df_copy = pd.concat([df_copy, workclass_dummy, education_dummy, maritalstatus_dummy, occupation_dummy, relationship_dummy, race_dummy, sex_dummy, nativecountry_dummy, income_dummy], axis=1)

# Delete unnecessary columns
del df_copy['workclass']
del df_copy['education']
del df_copy['marital-status']
del df_copy['occupation']
del df_copy['relationship']
del df_copy['race']
del df_copy['sex']
del df_copy['native-country']
del df_copy['income']

# %%

# DATA SPLITTING

X = df_copy.drop('income_>50K', axis=1)
y = df_copy['income_>50K']

X_train_valid, X_test ,y_train_valid, y_test = train_test_split(X,y, test_size = 0.2, random_state = 1000)

# %%

# Scaling only the numerical columns. Not scaling the dummy variables here due to ease of interpretation
numeric_cols = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
scaler = StandardScaler()
scaler.fit(X_train_valid[numeric_cols]) # Fit training data
X_train_valid.loc[:, numeric_cols] = scaler.transform(X_train_valid[numeric_cols]) # Transform (training+validation) data
X_test.loc[:, numeric_cols] = scaler.transform(X_test[numeric_cols]) # Transform testing data

# LOGISTIC REGRESSION USING STATSMODELS

X_train_valid_int = sm.add_constant(X_train_valid) # Adding the intercept manually
logit = sm.GLM(y_train_valid, X_train_valid_int, family = sm.families.Binomial()).fit()
logit.summary()

# %%

# Logistic Regression (using sklearn), a train-test split without threshold tuning (default threshold 0.5) and no penalty and 'newton-cg' solver
# METRICS

lr = LogisticRegression(penalty = 'none', solver = 'newton-cg', max_iter = 10000)
lr.fit(X_train_valid, y_train_valid)

y_pred = lr.predict(X_test)
print('\nLogistic Regression using sklearn and a train-test split without threshold tuning and newton-cg solver')
print('--------------------------------------------------------------------------------------------------------')
confmat = confusion_matrix(y_test, y_pred, labels = [1,0])
TP = confmat[0,0]
FN = confmat[0,1]
FP = confmat[1,0]
TN = confmat[1,1]

Accuracy = (TP + TN) / sum(sum(confmat)) #Accuracy
TPR = TP / (TP + FN) # True Positive Rate, Sensitivity, Recall
FPR = FP / (FP + TN) # False Positive Rate
TNR = TN / (TN + FP) # True Negative Rate, Specificity
FNR = FN / (FN + TP) # False Negative Rate
Precision = TP / (TP + FP) # Precision

print('Confusion Matrix for testing set\n', confmat)
print("False Positive Rate =", FPR)
print("False Negative Rate =", FNR)
print("True Positive Rate/Recall =", TPR)
print("True Negative Rate =", TNR)
#print("Accuracy = %f" %Accuracy)
print("Accuracy =", Accuracy)
print("Precision =", Precision)
print('F1 score of Testing Set =', f1_score(y_test, y_pred))

# %%

# Logistic Regression (using sklearn), 5-fold CV with threshold tuning and no penalty and 'newton-cg' solver

X_train_valid, X_test ,y_train_valid, y_test = train_test_split(X,y, test_size = 0.2, random_state = 1000)

thresholds = np.linspace(0, 1, 51) # Setting up a range of 50 threshold values
thresholds = np.delete(thresholds,[50])

F1_List = []

for ind, threshold in enumerate(thresholds): # Loop over each candidate value of thresholds
    kfold = KFold(n_splits = 5, shuffle = True, random_state = 1000) #Instantiating the fold
    
    temp_F1 = [] # To store the current threshold's F1-score for the current fold's data
    
    for train_index, valid_index in kfold.split(X_train_valid): # Splitting into Training and Validation set
        X_train, y_train = X_train_valid.iloc[train_index], y_train_valid.iloc[train_index] # Training set
        X_valid, y_valid = X_train_valid.iloc[valid_index], y_train_valid.iloc[valid_index] # Validation set
        
        #Scale the data
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = pd.DataFrame(scaler.transform(X_train))
        X_valid = pd.DataFrame(scaler.transform(X_valid))
        X_train.columns = X.columns.values
        X_valid.columns = X.columns.values
        
        #Perform Logistic Regression fitting on the training data without penalty, using newton-cg solver and max iterations of 10000
        lr = LogisticRegression(penalty = 'none', solver = 'newton-cg', max_iter = 10000)
        lr.fit(X_train, y_train)
        
        y_pred_threshold = np.where(lr.predict_proba(X_valid)[:,1] > threshold, 1, 0) # Getting predicted classes of the validation data for a particular threshold
        
        temp_F1.append(f1_score(y_valid, y_pred_threshold))
    
    F1_List.append(np.mean(temp_F1)) # Calculate the average F-1 score across all 5 folds       

# Saving threshold since we will use them later
best_threshold_maximizing_F1 = thresholds[np.argmax(F1_List)]

# Print Maximum F-1 score and its corresponding threshold
print('Logistic Regression with 5-fold CV - Validation Set Maximized F1 score is', max(F1_List), 'at a threshold value of', best_threshold_maximizing_F1)

# Again, scaling the  (training+validation) and the testing set
scaler = StandardScaler()
scaler.fit(X_train_valid)
X_train_valid = pd.DataFrame(scaler.transform(X_train_valid))
X_test = pd.DataFrame(scaler.transform(X_test))
X_train_valid.columns = X.columns.values
X_test.columns = X.columns.values

# Refitting the model with the (training + validation) data
lr.fit(X_train_valid, y_train_valid)

y_pred = np.where(lr.predict_proba(X_test)[:,1] > best_threshold_maximizing_F1, 1, 0) # Getting predicted classes of the testing data at our tuned threshold value
    
print('\nLogistic Regression using sklearn and a 5-fold CV with threshold tuning and newton-cg solver')
print('----------------------------------------------------------------------------------------------')
confmat = confusion_matrix(y_test, y_pred, labels = [1,0])
TP = confmat[0,0]
FN = confmat[0,1]
FP = confmat[1,0]
TN = confmat[1,1]

Accuracy = (TP + TN) / sum(sum(confmat)) #Accuracy
TPR = TP / (TP + FN) # True Positive Rate, Sensitivity, Recall
FPR = FP / (FP + TN) # False Positive Rate
TNR = TN / (TN + FP) # True Negative Rate, Specificity
FNR = FN / (FN + TP) # False Negative Rate
Precision = TP / (TP + FP) # Precision

print('Confusion Matrix for testing set\n', confmat)
print("False Positive Rate =", FPR)
print("False Negative Rate =", FNR)
print("True Positive Rate/Recall =", TPR)
print("True Negative Rate =", TNR)
#print("Accuracy = %f" %Accuracy)
print("Accuracy =", Accuracy)
print("Precision =", Precision)
print('F1 score of Testing Set =', f1_score(y_test, y_pred))

# %%

# Decision Tree, 5-fold CV with hyperparameter tuning (max_depth, max_leaf_nodes, and alpha)

X_train_valid, X_test ,y_train_valid, y_test = train_test_split(X,y, test_size = 0.2, random_state = 1000)

tree_grid = {'max_depth': np.arange(2,50,2),
             'max_leaf_nodes': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
             'criterion': ['gini', 'entropy']}

dt_clf = GridSearchCV(DecisionTreeClassifier(random_state = 1000, min_samples_leaf = 10), tree_grid, cv = 5, scoring = 'f1', n_jobs = -1)
dt_clf.fit(X_train_valid, y_train_valid)
y_pred = dt_clf.predict(X_test)

print('Decision Tree - Best Hyperparameters:', dt_clf.best_params_)
print('Decision Tree - F1 Score on Testing Set:', f1_score(y_test, y_pred))
print('Decision Tree - Confusion Matrix for Testing Set:\n', confusion_matrix(y_test, y_pred, labels = [1,0]))
print('Decision Tree - Accuracy of Testing Set:', accuracy_score(y_test, y_pred))

# Refit the model to get important features
refit_dt_clf = DecisionTreeClassifier(max_depth = dt_clf.best_params_['max_depth'], 
                             max_leaf_nodes = dt_clf.best_params_['max_leaf_nodes'],
                             criterion = dt_clf.best_params_['criterion'],
                             min_samples_leaf = 10, random_state = 1000)
refit_dt_clf.fit(X_train_valid, y_train_valid)
importance = pd.DataFrame({'feature':X.columns.values, 'importance':refit_dt_clf.feature_importances_})
print('Decision Tree - Features ranked according to their importance are as below: \n')
print(importance[importance['importance'] != 0].sort_values(by = ['importance'], ascending = False))

"""
Results that we got

Decision Tree - Best Hyperparameters: {'criterion': 'gini', 'max_depth': 24, 'max_leaf_nodes': 128}
Decision Tree - F1 Score on Testing Set: 0.6710050614605929
Decision Tree - Confusion Matrix for Testing Set:
 [[1392  890]
 [ 475 6278]]
Decision Tree - Accuracy of Testing Set: 0.8489208633093526
Decision Tree - Features ranked according to their importance are as below: 

                         feature  importance
14        marital-status_Married    0.351797
2                   capital-gain    0.232972
3                   capital-loss    0.101891
9            education_Bachelors    0.049664
0                            age    0.039830
12             education_Masters    0.039698
4                 hours-per-week    0.034175
17    occupation_Exec-managerial    0.028621
23     occupation_Prof-specialty    0.026032
8     education_Associate-degree    0.015725
32             relationship_Wife    0.012107
11    education_High-school-grad    0.011634
13         education_Prof-school    0.009389
29    relationship_Not-in-family    0.008566s
1                         fnlwgt    0.006428
28          relationship_Husband    0.006092
10           education_Doctorate    0.005852
21      occupation_Other-service    0.004759
18    occupation_Farming-fishing    0.003371
26       occupation_Tech-support    0.001937
6        workclass_Self-employed    0.001770
19  occupation_Handlers-cleaners    0.001531
5              workclass_Private    0.001192
36                    race_White    0.001171
15       occupation_Adm-clerical    0.001124
20  occupation_Machine-op-inspct    0.000802
38             native-country_US    0.000662
37                      sex_Male    0.000651
27   occupation_Transport-moving    0.000555
"""
# %%

# RandomForest, 5-fold CV with hyperparameter tuning (n_estimators, max_features, and min_samples_leaf) using RandomSearchCV

random_grid = {
    'n_estimators': np.arange(100, 1000, 100),
    'max_features': np.arange(1, 10, 1),
    'min_samples_leaf': np.arange(2, 10, 1),
    }

rf_clf_random = RandomizedSearchCV(RandomForestClassifier(random_state = 1000), param_distributions = random_grid, cv = 5, scoring = 'f1', n_jobs = -1, n_iter = 100)
rf_clf_random.fit(X_train_valid, y_train_valid)
rf_clf_random.best_params_

"""
We got the below best parameters

{'n_estimators': 600, 'min_samples_leaf': 2, 'max_features': 9}

"""


# %% 

# RandomForest, 5-fold CV with hyperparameter tuning (n_estimators, max_features, and min_samples_leaf) using GridSearchCV

rf_grid = {'n_estimators': np.arange(100, 500, 50),
           'max_features': np.arange(1, 7, 1),
           'min_samples_leaf': np.arange(2, 5, 1)}

rf_clf = GridSearchCV(RandomForestClassifier(random_state = 1000), param_grid = rf_grid, 
                   cv = 5, scoring = 'f1', n_jobs = -1)
rf_clf.fit(X_train_valid, y_train_valid)
y_pred = rf_clf.predict(X_test)

print('Random Forest - Best Hyperparameters:', rf_clf.best_params_)
print('Random Forest - F1 Score on Testing Set:', f1_score(y_test, y_pred))
print('Random Forest - Confusion Matrix for Testing Set:\n', confusion_matrix(y_test, y_pred, labels = [1,0]))
print('Random Forest - Accuracy of Testing Set:', accuracy_score(y_test, y_pred))

# Refit the model to get important features
refit_rf_clf = RandomForestClassifier(n_estimators = rf_clf.best_params_['n_estimators'], 
                                      max_features = rf_clf.best_params_['max_features'],
                                      min_samples_leaf = rf_clf.best_params_['min_samples_leaf'], random_state = 1000)
refit_rf_clf.fit(X_train_valid, y_train_valid)
importance = pd.DataFrame({'feature':X.columns.values, 'importance':refit_rf_clf.feature_importances_})
print('Random Forest - Features ranked according to their importance are as below: \n')
print(importance[importance['importance'] != 0].sort_values(by = ['importance'], ascending = False))

"""
Results that we got

Random Forest - Best Hyperparameters: {'max_features': 6, 'min_samples_leaf': 3, 'n_estimators': 450}
Random Forest - F1 Score on Testing Set: 0.6736214605067065
Random Forest - Confusion Matrix for Testing Set:
 [[1356  926]
 [ 388 6365]]
Random Forest - Accuracy of Testing Set: 0.8545655783065855
Random Forest - Features ranked according to their importance are as below: 

                         feature  importance
2                   capital-gain    0.167997
14        marital-status_Married    0.112531
0                            age    0.110246
28          relationship_Husband    0.098998
1                         fnlwgt    0.070412
4                 hours-per-week    0.067252
3                   capital-loss    0.047863
17    occupation_Exec-managerial    0.035258
9            education_Bachelors    0.034523
23     occupation_Prof-specialty    0.032433
32             relationship_Wife    0.025161
12             education_Masters    0.021008
29    relationship_Not-in-family    0.019707
37                      sex_Male    0.016669
11    education_High-school-grad    0.016064
31        relationship_Own-child    0.015121
13         education_Prof-school    0.012910
8     education_Associate-degree    0.012852
21      occupation_Other-service    0.011225
5              workclass_Private    0.008879
10           education_Doctorate    0.007342
6        workclass_Self-employed    0.006519
25              occupation_Sales    0.006384
18    occupation_Farming-fishing    0.005955
38             native-country_US    0.004901
26       occupation_Tech-support    0.004736
15       occupation_Adm-clerical    0.004123
20  occupation_Machine-op-inspct    0.004067
36                    race_White    0.004056
27   occupation_Transport-moving    0.003384
19  occupation_Handlers-cleaners    0.003335
35                    race_Black    0.002890
24    occupation_Protective-serv    0.002234
34       race_Asian-Pac-Islander    0.001557
30   relationship_Other-relative    0.000889
33       race_Amer-Indian-Eskimo    0.000466
22    occupation_Priv-house-serv    0.000042
7          workclass_Without-pay    0.000009
16       occupation_Armed-Forces    0.000002
"""

# %%

# GradientBoosting 5-fold CV with hyperparameter tuning (n_estimators, max_features, and learning_rate)


gb_grid = {'n_estimators': [100,200,300],
           'learning_rate': [0.01, 0.05, 0.1],
           'max_depth': [2,3,4]}
gb_clf = GridSearchCV(GradientBoostingClassifier(min_samples_leaf = 5, random_state = 1000),
                    param_grid = gb_grid, cv = 5, n_jobs = -1, scoring = 'f1')
gb_clf.fit(X_train_valid, y_train_valid)
y_hat = gb_clf.predict(X_test)

print('Gradient Boosting - Best Hyperparameters:', gb_clf.best_params_)
print('Gradient Boosting - F1 Score on Testing Set:', f1_score(y_test, y_hat))
print('Gradient Boosting - Confusion Matrix for Testing Set:\n', confusion_matrix(y_test, y_hat, labels = [1,0]))
print('Gradient Boosting - Accuracy of Testing Set:', accuracy_score(y_test, y_hat))

# Refit the model to get important features
refit_gb_clf = GradientBoostingClassifier(n_estimators = gb_clf.best_params_['n_estimators'], 
                                          learning_rate = gb_clf.best_params_['learning_rate'],
                                          max_depth = gb_clf.best_params_['max_depth'],
                                          min_samples_leaf = 5, random_state = 1000)
refit_gb_clf.fit(X_train_valid, y_train_valid)
importance3 = pd.DataFrame({'feature':X.columns.values, 'importance':refit_gb_clf.feature_importances_})
print('Boosting - Features ranked according to their importance are as below: \n')
print(importance3[importance3['importance'] != 0].sort_values(by = ['importance'], ascending = False))

"""
Results that we got

Gradient Boosting - Best Hyperparameters: {'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 300}
Gradient Boosting - F1 Score on Testing Set: 0.7048501096758469
Gradient Boosting - Confusion Matrix for Testing Set:
 [[1446  836]
 [ 375 6378]]
Gradient Boosting - Accuracy of Testing Set: 0.8659656889872718
Boosting - Features ranked according to their importance are as below: 

                         feature  importance
2                   capital-gain    0.231591
28          relationship_Husband    0.196180
14        marital-status_Married    0.122277
3                   capital-loss    0.077494
0                            age    0.063394
4                 hours-per-week    0.047945
32             relationship_Wife    0.044703
23     occupation_Prof-specialty    0.036462
9            education_Bachelors    0.035509
17    occupation_Exec-managerial    0.031691
1                         fnlwgt    0.020606
12             education_Masters    0.019430
13         education_Prof-school    0.011141
21      occupation_Other-service    0.009992
10           education_Doctorate    0.008080
11    education_High-school-grad    0.007285
18    occupation_Farming-fishing    0.006115
8     education_Associate-degree    0.005528
26       occupation_Tech-support    0.003626
25              occupation_Sales    0.003429
5              workclass_Private    0.002320
20  occupation_Machine-op-inspct    0.002188
37                      sex_Male    0.002117
6        workclass_Self-employed    0.001946
29    relationship_Not-in-family    0.001752
19  occupation_Handlers-cleaners    0.001559
24    occupation_Protective-serv    0.001178
38             native-country_US    0.000994
15       occupation_Adm-clerical    0.000658
27   occupation_Transport-moving    0.000557
36                    race_White    0.000551
34       race_Asian-Pac-Islander    0.000521
35                    race_Black    0.000382
31        relationship_Own-child    0.000300
33       race_Amer-Indian-Eskimo    0.000281
30   relationship_Other-relative    0.000131
7          workclass_Without-pay    0.000055
16       occupation_Armed-Forces    0.000018
22    occupation_Priv-house-serv    0.000012 """


