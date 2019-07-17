# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('titanic_data.csv')
dataset.drop(['PassengerId','Name','Cabin','Embarked','Ticket','Fare'],axis=1, inplace=True)
dataset.dropna(inplace=True)

#Initialise the dependent and independent variable
X = dataset.iloc[:, 1:6].values
y = dataset.iloc[:, 0].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)


X_test_real = X_test  
X_test_real = X_test_real.astype(str)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X_train[:,1] = labelencoder_X.fit_transform(X_train[:,1])
X_test[:,1] = labelencoder_X.fit_transform(X_test[:,1])

onehotencoder = OneHotEncoder(categorical_features=[0])
X_train = onehotencoder.fit_transform(X_train).toarray()
X_test = onehotencoder.fit_transform(X_test).toarray()


X_train = X_train[:,1:] #Dummy variable trap

X_test = X_test[:,1:] #Dummy variable trap


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Fitting data into logistic regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix (For result)
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test, y_pred)

#calculate accuracy
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)




