import pandas as pd


dataset_kocak = pd.read_csv("./r2_gse62564_GSVA_Metadata_selected.csv")
dataset_kocak.head(10)


X = dataset_kocak.iloc[:, 1:-1].values
y = dataset_kocak.iloc[:, -1]


########## Creating the Training Set and the Test Set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

########## Feature Scaling the Feature Array

from sklearn.preprocessing import StandardScaler
# Standardization based on the features of the whole dataset (?)
# Compute in the training set (?)
# instance of the class
sc = StandardScaler()
# compute average and sd of the features
# Takes on the array of independent variables you want to scale
sc.fit_transform(X_train)
# We will only need the new array of independent variables in the training set
X_train = sc.fit_transform(X_train)

###################################################
########## Part 2 - Building and training the model
###################################################

########## Import library
from sklearn.linear_model import LogisticRegression


########## Building the model
model = LogisticRegression(random_state = 0 )


########## Training the model
model.fit(X_train, y_train)


########## Access coefficients for variable importance
coefficients = model.coef_[0]


########## Plot variable importance

X_test_kocak = dataset_kocak.drop('high_risk', axis=1)
X_test_kocak = X_test_kocak.drop('sample id', axis=1)


########## Plot variable importance

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

feature_importance = pd.DataFrame({'Feature': X_test_kocak.columns, 'Importance': np.abs(coefficients)})
# feature_importance = feature_importance.sort_values('Importance', ascending=True).head(70)
feature_importance = feature_importance.sort_values('Importance', ascending=True)
# feature_importance = feature_importance[:5000]
feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6))

########## Inference

# Predictions for the test set and for a particular patient
y_pred = model.predict(sc.transform(X_test)) # First, call the scaler object

input_string = input('Enter elements of a list separated by space \n')
user_list = input_string.split()
print('User list: ', user_list)
model.predict(sc.transform([user_list]))