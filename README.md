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
import matplotlib.pyplot as plt
df=pd.read_csv('/content/titanic_dataset.csv')
df.head()
df.isnull().sum()
df.drop('Cabin',axis=1,inplace=True)
df.drop('Name',axis=1,inplace=True)
df.drop('Ticket',axis=1,inplace=True)
df.drop('PassengerId',axis=1,inplace=True)
df.drop('Parch',axis=1,inplace=True)
df.head()
df['Age']=df['Age'].fillna(df['Age'].median())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
df.isnull().sum()
plt.title("Dataset with outliers")
df.boxplot()
plt.show()
cols = ['Age','SibSp','Fare']
Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
plt.title("Dataset after removing outliers")
df.boxplot()
plt.show()
from sklearn.preprocessing import OrdinalEncoder
climate = ['C','S','Q']
en= OrdinalEncoder(categories = [climate])
df['Embarked']=en.fit_transform(df[["Embarked"]])
df.head()
from sklearn.preprocessing import OrdinalEncoder
gender = ['male','female']
en= OrdinalEncoder(categories = [gender])
df['Sex']=en.fit_transform(df[["Sex"]])
df.head()
from sklearn.preprocessing import RobustScaler
sc=RobustScaler()
df=pd.DataFrame(sc.fit_transform(df),columns=['Survived','Pclass','Sex','Age','SibSp','Fare','Embarked'])
df.head()
import statsmodels.api as sm
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal',n_quantiles=692)

df1=pd.DataFrame()
df1["Survived"]=np.sqrt(df["Survived"])
df1["Pclass"],parameters=stats.yeojohnson(df["Pclass"])
df1["Sex"]=np.sqrt(df["Sex"])
df1["Age"]=df["Age"]
df1["SibSp"],parameters=stats.yeojohnson(df["SibSp"])
df1["Fare"],parameters=stats.yeojohnson(df["Fare"])
df1["Embarked"]=df["Embarked"]
df1.skew()
import matplotlib
import seaborn as sns
import statsmodels.api as sm
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

X = df1.drop("Survived",1) 
y = df1["Survived"]
plt.figure(figsize=(7,6))
cor = df1.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.RdPu)
plt.show()
cor_target = abs(cor["Survived"])
relevant_features = cor_target[cor_target>0.5]
relevant_features
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)
nof_list=np.arange(1,6)            
high_score=0
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,step=nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))
cols = list(X.columns)
model = LinearRegression()
rfe = RFE(model, step=2)             
X_rfe = rfe.fit_transform(X,y)  
model.fit(X_rfe,y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)
reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (5.0, 5.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
plt.show()
```

# OUTPUT

## DATA PREPROCESSING BEFORE FEATURE SELECTION:
<img width="614" alt="s1" src="https://github.com/SmritiManikand/ODD2023-Datascience-Ex-07/assets/113674204/ac1acf6d-505c-4ada-bee7-aede88bd4a12">

## CHECKING NULL VALUES:
<img width="217" alt="s2" src="https://github.com/SmritiManikand/ODD2023-Datascience-Ex-07/assets/113674204/67cffe21-f016-4c09-8fb6-311b1ed1e421">

## DROPPING UNWANTED DATAS:
<img width="425" alt="s3" src="https://github.com/SmritiManikand/ODD2023-Datascience-Ex-07/assets/113674204/683cf4d8-bc3e-4ec4-a360-b7f19db4aac6">

## DATA CLEANING:
<img width="191" alt="s4" src="https://github.com/SmritiManikand/ODD2023-Datascience-Ex-07/assets/113674204/2c2c8477-12f1-4f6d-a813-cb6a2f6e4eb5">

## BEFORE REMOVING OUTLIERS:
<img width="233" alt="s5" src="https://github.com/SmritiManikand/ODD2023-Datascience-Ex-07/assets/113674204/cb0cc959-04b5-4c5c-b668-8fb15b9207c9">

## AFTER REMOVING OUTLIERS:
<img width="230" alt="s6" src="https://github.com/SmritiManikand/ODD2023-Datascience-Ex-07/assets/113674204/ef97a214-747f-48a3-b387-b877ed376df2">

## FEATURE SELECTION:
<img width="420" alt="s7" src="https://github.com/SmritiManikand/ODD2023-Datascience-Ex-07/assets/113674204/819dbfd8-f1f4-40ad-8d8a-26b648113a21">

<img width="433" alt="s8" src="https://github.com/SmritiManikand/ODD2023-Datascience-Ex-07/assets/113674204/2d22c6df-a3a1-4742-b6cc-58e1f7c23f17">

<img width="428" alt="s9" src="https://github.com/SmritiManikand/ODD2023-Datascience-Ex-07/assets/113674204/642cc858-2519-42cf-a41e-80b97e44ca4f">

## FILTER METHOD:
<img width="230" alt="s10" src="https://github.com/SmritiManikand/ODD2023-Datascience-Ex-07/assets/113674204/bad5db0e-50d2-4b34-8b35-008af5867692">

## HIGHLY CORRELATED FEATURES WITH THE OUTPUT VARIABLE SURVIVED:
<img width="237" alt="s11" src="https://github.com/SmritiManikand/ODD2023-Datascience-Ex-07/assets/113674204/ddc8dcfd-1ba2-4958-8a04-1ad72153569f">

## BACKWARD ELIMINATION:
<img width="305" alt="s12" src="https://github.com/SmritiManikand/ODD2023-Datascience-Ex-07/assets/113674204/e28a9a22-13fc-4739-a5cb-1d8b33f58ea0">

## OPTIMUM NUMBER OF FEATURES THAT HAVE HIGH ACCURACY:
<img width="273" alt="s13" src="https://github.com/SmritiManikand/ODD2023-Datascience-Ex-07/assets/113674204/6dcf4284-e825-4687-b4f8-462bede267e6">

## FINAL SET OF FEATURE:
<img width="359" alt="s14" src="https://github.com/SmritiManikand/ODD2023-Datascience-Ex-07/assets/113674204/bf1935c3-bc49-46a9-98d8-eb59cedc5b93">

## EMBEDDED METHOD:
<img width="232" alt="s15" src="https://github.com/SmritiManikand/ODD2023-Datascience-Ex-07/assets/113674204/9a1dc855-1f68-497d-a65a-72bb030c9763">

# RESULT
Thus, the various feature selection techniques have been performed on a given dataset successfully.
