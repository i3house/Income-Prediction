# income-classification
This was the project for our Data Mining and Advanced Statistical Methods for Data Analysis course at SF State University. The dataset used was the [1994 Census Income](https://archive.ics.uci.edu/dataset/2/adult) dataset from the UCI repository. The data files are also attached here. 

About the data: Each unit of observation belongs to a particular individual (48,842 entries) and consists of categorical data (workclass, education, marital status, occupation, relationship, race, sex, native country, and income) and numerical data (age, number of years of education, capital gain, capital loss, hours per week)
More information about the dataset it present in the final report and the presentation!

The variable income is the response variable.

Objective: To predict whether an individual's income will be greater than $50K (A binary classification problem).

We preprocessed the data and developed understanding of the data by doing exploratory analysis. We then implememented several classification models (Logistic Regression, Decision Trees, Random Forest Classifier, and Gradient Boosting Classifier), tuned their hyperparameters and finally compared their results on the training and testing set to arrive at a model that works best on both test and training data. We used F1 score as the metric because the response variable was imbalaned
