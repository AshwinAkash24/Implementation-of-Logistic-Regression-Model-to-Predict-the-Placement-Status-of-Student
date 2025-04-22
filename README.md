# EX-05:Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## AlgorithmLoad the data from CSV and remove extra column
#### STEP1: Load the csv file
(We use pandas to read the file and drop the serial number which isn't useful.)

#### STEP2: Convert text data to numbers
(Machine learning models need numbers, not text, so we encode categorical data.)

#### Step3: Split data into input (X) and output (y)
(X has all features, y has what we want to predict – placement status.)

#### Step4: Split data into training and testing sets
(We train the model on one part and test it on another to see how well it works.)

#### Step5: Train the logistic regression model
(We use a simple model to learn patterns from the training data.)

#### Step6: Make predictions and check accuracy
(We test the model with new data and measure how accurate it is.)

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Ashwin Akash M
RegisterNumber:  212223230024
/*

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("Placement_Data.csv")
df
df.info()
df=df.drop('sl_no',axis=1)
df
df["gender"]=df["gender"].astype('category')
df["ssc_b"]=df["ssc_b"].astype('category')
df["hsc_b"]=df["hsc_b"].astype('category')
df["degree_t"]=df["degree_t"].astype('category')
df["workex"]=df["workex"].astype('category')
df["specialisation"]=df["specialisation"].astype('category')
df["status"]=df["status"].astype('category')
df["hsc_s"]=df["hsc_s"].astype('category')
df.dtypes
df.info()
df["gender"]=df["gender"].cat.codes
df["ssc_b"]=df["ssc_b"].cat.codes
df["hsc_b"]=df["hsc_b"].cat.codes
df["degree_t"]=df["degree_t"].cat.codes
df["workex"]=df["workex"].cat.codes
df["specialisation"]=df["specialisation"].cat.codes
df["status"]=df["status"].cat.codes
df["hsc_s"]=df["hsc_s"].cat.codes
df=df.drop('salary',axis=1)
df
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
x
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3)
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(max_iter=10000)
clf.fit(x_train,y_train)
clf.score(x_test,y_test)
ypred=clf.predict(x_test)
ypred
from sklearn.metrics import accuracy_score
acc=accuracy_score(ypred,y_test)
acc
clf.predict([[0,87,0,95,0,2,78,2,0,0,1,0]])
clf.predict([[0,0,1,92,1,2,46,1,0,1,0,1]])
```

## Output:
![image](https://github.com/user-attachments/assets/91db7c1f-f8aa-4db9-83a5-ad25c7119470)<br>
![image](https://github.com/user-attachments/assets/a3a7ddf5-0a17-4e64-9155-96333dca4cbe)<br>
![image](https://github.com/user-attachments/assets/1222efbd-27e5-4423-bd20-83711608a18a)<br>
![image](https://github.com/user-attachments/assets/80c05b6a-e417-4e7a-9ebc-360c691a4170)<br>
![image](https://github.com/user-attachments/assets/a033275d-17f6-4da0-85f9-caee97b94ff4)<br>
![image](https://github.com/user-attachments/assets/8d3ba2d2-c9a0-44bb-805e-bb5c55c03714)<br>
![image](https://github.com/user-attachments/assets/56c7eed8-7abc-45b9-ae0d-4bb1ed9c1257)<br>
![image](https://github.com/user-attachments/assets/1dc7469d-9cf0-4916-9676-a260f7fcb399)<br>
![image](https://github.com/user-attachments/assets/44504747-04f6-4004-83a4-7c8bf35265d5)<br>
![image](https://github.com/user-attachments/assets/04303f8d-d58e-4f20-ba50-e56fa253f978)<br>
![image](https://github.com/user-attachments/assets/8d4788c4-b90e-4270-b0af-ed7b5cf3b32f)<br>
![image](https://github.com/user-attachments/assets/a5cd90dc-f334-4334-9296-cbd79aefd284)<br>
![image](https://github.com/user-attachments/assets/8371cde5-ef7e-4a80-87a0-71cf6855edaf)<br>


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
