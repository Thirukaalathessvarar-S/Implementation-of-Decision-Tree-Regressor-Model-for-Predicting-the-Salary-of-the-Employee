# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null values using .isnull() function.
3. Import LabelEncoder and encode the dataset.
4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5. Predict the values of arrays.
6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7. Predict the values of array.
8. Apply to new unknown values.

## Program:
```

Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Thirukaalathessvarar S
RegisterNumber:  212222230161

```
```
import pandas as pd
data=pd.read_csv("/content/Salary.csv")
data.head()

```
```
data.info()
```
```
data.isnull().sum()
```
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['Position']=le.fit_transform(data['Position'])
data.head()
```
```
x=data[['Position','Level']]
x
```
```
y=data['Salary']
y
```
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
```
```
from sklearn.tree import DecisionTreeClassifier,plot_tree
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

```
```
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
```
```
r2=metrics.r2_score(y_test,y_pred)
r2
```
```
import matplotlib.pyplot as plt
dt.predict([[5,6]])
plt.figure(figsize=(20,8))
plot_tree(dt,feature_names=x.columns,filled=True)
plt.show()
```
## Output:
### Head()
![image](https://github.com/Jayakrishnan22003251/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120232371/a38d62f9-e669-4eab-be28-a694eb0837ff)
### Dataset Info()
![image](https://github.com/Jayakrishnan22003251/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120232371/b1d3986e-bdb1-40ec-b708-18b8b3944280)
### Null Values()
![image](https://github.com/Jayakrishnan22003251/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120232371/85f2d56f-8719-4abc-832d-f0eccfa71aff)
### Label Encoding()
![image](https://github.com/Jayakrishnan22003251/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120232371/cea1c385-1f93-47bc-9562-4dfec6cfb470)
### Splitting dataset for Dependent and Independent Variable()
![image](https://github.com/Jayakrishnan22003251/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120232371/eec81356-d2b4-4c1c-82c5-bee3751facce)

![image](https://github.com/Jayakrishnan22003251/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120232371/b9389759-f08f-4399-8592-3f59305d7f35)
### MSE Value()
![image](https://github.com/Jayakrishnan22003251/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120232371/112f3e47-9deb-421c-900f-cd554a9c3cbf)
### R2 Value()
![image](https://github.com/Jayakrishnan22003251/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120232371/703bb0f8-7f24-4571-944d-077bc976017c)
### Plot()
![image](https://github.com/Jayakrishnan22003251/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120232371/5cf8b143-af2e-4f7c-8880-8b1365c422cb)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
