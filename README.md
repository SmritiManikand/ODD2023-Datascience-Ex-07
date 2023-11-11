# Ex-07-Feature-Selection
# AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# EXPLANATION
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file


# CODE

```
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
df=pd.read_csv("/content/titanic_dataset.csv")
df.columns

df.shape

X=df.drop("Survived",1)
y=df['Survived']

df1=df.drop(['Name','Sex','Ticket','Cabin','Embarked'],axis=1)
df1.columns

df1['Age'].isnull().sum()

df1['Age'].fillna(method='ffill')

df1['Age']=df1['Age'].fillna(method='ffill')

df1['Age'].isnull().sum()

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
data=pd.read_csv("/content/titanic_dataset.csv")
data=data.dropna()

X=data.drop(["Survived","Name","Ticket"],axis=1)
y=data['Survived']
X

data['Sex']=data['Sex'].astype('category')
data['Cabin']=data['Cabin'].astype('category')
data['Embarked']=data['Embarked'].astype('category')

data['Sex']=data['Sex'].cat.codes
data['Cabin']=data['Cabin'].cat.codes
data['Embarked']=data['Embarked'].cat.codes

data

k=5
selector=SelectKBest(score_func=chi2,k=k)
X_new=selector.fit_transform(X,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=X.columns[selected_feature_indices]
print(selected_features)

from sklearn.feature_selection import SelectKBest,f_regression
selector=SelectKBest(score_func=f_regression,k=5)
X_new=selector.fit_transform(X,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=X.columns[selected_feature_indices]
print(selected_features)

from sklearn.feature_selection import SelectKBest,mutual_info_classif
selector=SelectKBest(score_func=mutual_info_classif,k=5)
X_new=selector.fit_transform(X,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=X.columns[selected_feature_indices]
print(selected_features)

import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
sfm=SelectFromModel(model,threshold='mean')
sfm.fit(X,y)
selected_features=X.columns[sfm.get_support()]
print(selected_features)

import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
num_features_to_remove=2
rfe=RFE(model,n_features_to_select=(len(X.columns)-num_features_to_remove))
rfe.fit(X,y)
selected_features=X.columns[rfe.support_]
print(selected_features)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlxtend.feature_selection import ExhastiveFeatureSelector
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=RandomForestClassifier(n_estimators=100,random_state=42)
efs=ExhaustiveFeatureSelector(model,min_features=1,max_features=len(X.columns),scoring='accuracy',cv=5)
efs=efs.fit(X_train,y_train)
selected_features=list(X.columns[list(efs.best_idex_)])
model.fit(X_train[selected_features],y_train)
y_pred=model.predict(X_test[selected_features])
accuracy=accuracy_score(y_test,y_pred)
print(selected_features)
print(accuracy)

import pandas as pd
from sklearn.linear_model import Lasso
model=Lasso(alpha=0.01)
model.fit(X,y)
feature_coefficients=model.coef_
selected_features=X.colums[feature_coefficients!=0]
print(selected_features)

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X,y)
feature_importances=model.feature_importances_
threshold=0.15
selected_features=X.columns[feature_importances>threshold]
print(selected_features)
```

# OUTPUT

## Data Cleaning and Processing

## Filter Method
### Chi2 Method

## Correlation Coefficient

## Mutual Information

## Wrapper Method Forward Selection

## Backward Elimination

## Exhaustive Feature Selection

## Embedded Methods

# RESULT
Thus, the various feature selection techniques have been performed on a given dataset successfully.
