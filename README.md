# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
```
```
df=pd.read_csv('/content/bmi.csv')
df
```

![image](https://github.com/user-attachments/assets/f385ef38-4813-4b79-9fc3-c70236d5bd9c)


```
df.head()
```

![image](https://github.com/user-attachments/assets/99288148-e80d-480d-978f-da76b6c7d352)


```
df.dropna()
```
![image](https://github.com/user-attachments/assets/d86047d4-9197-4580-bf99-3753c4a6b628)


```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
199
```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```

![image](https://github.com/user-attachments/assets/2394f3d4-96c5-4d51-8891-503d23b984e4)


```
df1=pd.read_csv('/content/bmi.csv')
df2=pd.read_csv('/content/bmi.csv')
df3=pd.read_csv('/content/bmi.csv')
df4=pd.read_csv('/content/bmi.csv')
df5=pd.read_csv('/content/bmi.csv')
df1
```

![image](https://github.com/user-attachments/assets/3c88eba3-90b8-4c3f-9555-ccfc8ba760f9)


```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/42bfe116-828f-4ceb-8be7-7c4e05e26753)


```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df2[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
df2
```

![image](https://github.com/user-attachments/assets/6c26ef3e-079c-488a-b4b1-534795292707)


```
from sklearn.preprocessing import MaxAbsScaler
max1=MaxAbsScaler()
df3[['Height','weight']]=max1.fit_transform(df3[['Height','Weight']])
df3
```
![image](https://github.com/user-attachments/assets/83a19fee-d10b-49bb-baa5-da421a2585c7)


```
from sklearn.preprocessing import RobustScaler
roub=RobustScaler()
df4[['Height','Weight']]=roub.fit_transform(df4[['Height','Weight']])
df4
```

![image](https://github.com/user-attachments/assets/d9384b58-812e-4b02-89d6-1c74e60eb92a)


```
from sklearn.feature_selection import SelectKBest,f_regression,mutual_info_classif
from sklearn.feature_selection import chi2
data=pd.read_csv('/content/income.csv')
data
```

![image](https://github.com/user-attachments/assets/a746855b-fabc-4c88-9abc-cfc759e64587)


```
data1=pd.read_csv('/content/titanic_dataset (1).csv')
data1
```

![image](https://github.com/user-attachments/assets/40be0a5f-d7fc-46d1-8ce8-4e37a8bab56b)




```
data1=data1.dropna()
x=data1.drop(['Survived','Name','Ticket'],axis=1)
y=data1['Survived']
data1['Sex']=data1['Sex'].astype('category')
data1['Cabin']=data1['Cabin'].astype('category')
data1['Embarked']=data1['Embarked'].astype('category')
```
```
data1['Sex']=data1['Sex'].cat.codes
data1['Cabin']=data1['Cabin'].cat.codes
data1['Embarked']=data1['Embarked'].cat.codes
```
```
data1
```

![image](https://github.com/user-attachments/assets/03996577-41bb-4412-b31f-14a9e9c31ae7)



```
k=5
selector=SelectKBest(score_func=chi2,k=k)
x=pd.get_dummies(x)
x_new=selector.fit_transform(x,y)
```
```
x_encoded=pd.get_dummies(x)
selector=SelectKBest(score_func=chi2,k=5)
x_new=selector.fit_transform(x_encoded,y)
```
```
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```

![image](https://github.com/user-attachments/assets/d03e6b8e-25f9-4562-85b1-c32261ab1652)


```
selector=SelectKBest(score_func=f_regression,k=5)
x_new=selector.fit_transform(x_encoded,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```


![image](https://github.com/user-attachments/assets/42a89e33-d693-4945-b001-e472692a70ee)

```
selector=SelectKBest(score_func=mutual_info_classif,k=5)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/a5ef2ad2-d54e-460f-956a-d95b9f9e1a7b)


```
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
```
```
model=RandomForestClassifier()
sfm=SelectFromModel(model,threshold='mean')
x=pd.get_dummies(x)
sfm.fit(x,y)
selected_features=x.columns[sfm.get_support()]
print("Selected Features:")
print(selected_features)
```

![image](https://github.com/user-attachments/assets/d024fddb-3894-49c1-8a7b-c512d6f3a22a)


```
from sklearn.ensemble import RandomForestClassifier
```
```
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x,y)
feature_selection=model.feature_importances_
threshold=0.1
selected_features=x.columns[feature_selection>threshold]
print("Selected Features:")
print(selected_features)
```

![image](https://github.com/user-attachments/assets/4c002a38-2b05-4764-bb85-e12acd17e8fc)


```
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x,y)
feature_importance=model.feature_importances_
threshold=0.15
selected_features=x.columns[feature_importance>threshold]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/115b5af5-5ec2-42b3-993f-e81e14e6b70d)




# RESULT:
Thus the feature selection and feature scaling has been used on the given dataset and executed successfully.
