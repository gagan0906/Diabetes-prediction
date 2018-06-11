# Diabetes-prediction
A Machine learning Classification model in python to detect if a female is suffering from diabetes.

## Data
Dataset is from Pima Indians Diabetes Database, taken from [Kaggle.com](https://www.kaggle.com/futurist/pima-data-visualisation-and-machine-learning/data).<br><br>
Dataset Contains number of pregnancies the patient has had, their BMI, insulin level, age, Blood pressure, Skin Thikness and the outcome(if diabetes is existing or not).<br><br>
We will be in need of preprocessing the data, as the whole data is containg some of the missing values.

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
This problem can be solved in analysis where we can simply think that, 'skin thikness' will not effect in diabetes, and we will remove that feature.<br><br>
Other option is **Feature_selection** from **SKLearn** Library.<br>
We need to specify the k i.e. number of predictors required, which shouid be taken good care, as it will affect the accuracy of our model.<br>
If we select a less values of k, then there may be a chance of overfitting, so we need to specify it as required by the use case.
>from sklearn.feature_selection import SelectKBest<br>
>from sklearn.feature_selection import chi2<br>
>X = SelectKBest(chi2, k=4).fit_transform(X, y)<br><br>
So, After this, we got 4 most effective features from the feature matrix, they are Age,BMI, Insulin, Glucose.
## Managing Missing Values
The dataset contain missing values, which needs to be managed as we know that BMI of any person cannot be 0.<br>
So we use **Sklearn.preprocessing** library to get **Imputer** class, which statisticaly manages missing values<br>
As, it is a medical data, missing data should be replaced be the thresold value of the feature, or it can be handled by general method, of taking mean of the whole column.<br>
>from sklearn.preprocessing import Imputer<br>
>imputer = Imputer(missing_values=0 , strategy="mean",axis=0)<br>
>X[:,0:5] = imputer.fit_transform(X[:,0:5])<br>

## Splitting the dataset into the Training set and Test set
This a part of Cross Validating our trained model with known values.<br>
taking 25% of the data as the test data. Which will be used to cross validate our trained data for accuracy<br>
>from sklearn.cross_validation import train_test_split<br>
>X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)<br>
## Creating the Classification Model
I am using K-neareast neighbours algorithm to classify in either having diabetes or not.<br>
You can use any of the classification model, but it will affect the accuracy(i'm not saying KNN is the most efficient model in this problem), so use any classifier as you like.<br>
We can even use Neural Networks to make a classificaion model like this, but its accuracy will be different.

>from sklearn.neighbors import KNeighborsClassifier<br>
>classifier = KNeighborsClassifier(n_neighbors = 5,metric = 'minkowski',p=2)<br>
>classifier.fit(X_train,y_train)<br>

## Predicting the Test set results
Getting a predicted outcome for the known test data, that we have in for cross validation.
>y_pred = classifier.predict(X_test)<br>

## Making the Confusion Matrix
A confusion matrix is a table that is often used to describe the performance of a classification model (or "classifier") on a set of test data for which the true values are known.<br>
>from sklearn.metrics import confusion_matrix<br>
>cm = confusion_matrix(y_test, y_pred)<br>
<img src="https://github.com/ujjwalanand1997/Diabetes-prediction/blob/master/confusion_matrix.png?raw=true" width="300"><br>
## Accuracy 
The confusion matrix is giving the measurement of accuracy of the model, In Which the correct values are 156(115+41) out of 192 values.<br>
Hence, Accuracy outcome is 81.25%.<br>
The above accuracy is calculated from the cross validation of test data<br>


