# Diabetes-prediction
A Machine learning Classification model in python to detect if a female is suffering from diabetes.

## Data
Dataset is from Pima Indians Diabetes Database, taken from [Kaggle.com](https://www.kaggle.com/futurist/pima-data-visualisation-and-machine-learning/data).<br><br>
Dataset Contains number of pregnancies the patient has had, their BMI, insulin level, age, Blood pressure, Skin Thikness and the outcome(if diabetes is existing or not).

## Importing Libraries

>import numpy as np<br>
>import pandas as pd<br>

## Selecting features to be used in making the model
As it is a classification problem to classify if a person is suffering from diabetes or not, So, We take 'outcome' as target and all other variables(BMI,no of pregnancies, etc) as predictors.<br>
>dataset = pd.read_csv('diabetes.csv')<br>
>X = dataset.iloc[:, 0:8].values<br>
>y = dataset.iloc[:, 8].values<br>

### Feature Selection
Selection of the effective features in predictors to make an accurate model.<br>
This problem can be solved in analysis where we can simply think that, 'skin thikness' will not effect in diabetes.<br><br>
Other option is **Feature_selection** from **SKLearn** Library.<br>
We need to specify the k i.e. number of predictors required, which shouid be taken good care, as it will affect the accuracy of our model.<br>
>from sklearn.feature_selection import SelectKBest<br>
>from sklearn.feature_selection import chi2<br>
>X = SelectKBest(chi2, k=4).fit_transform(X, y)<br><br>

