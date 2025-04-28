# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Load the dataset, drop unnecessary columns, and encode categorical variables. 
2. Define the features (X) and target variable (y). 
3. Split the data into training and testing sets. 
4. Train the logistic regression model, make predictions, and evaluate using accuracy and other
```

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Rakesh rathna M
RegisterNumber:  212224040265
*/
```
```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv(r"C:\Users\admin\Desktop\Placement_Data.csv")
data.head()

datal = data.copy()
datal = datal.drop(["sl_no", "salary"], axis=1)
datal.head()

print("Missing values:\n", datal.isnull().sum())
print("Duplicate rows:", datal.duplicated().sum())

le = LabelEncoder()
datal["gender"] = le.fit_transform(datal["gender"])
datal["ssc_b"] = le.fit_transform(datal["ssc_b"])
datal["hsc_b"] = le.fit_transform(datal["hsc_b"])
datal["hsc_s"] = le.fit_transform(datal["hsc_s"])
datal["degree_t"] = le.fit_transform(datal["degree_t"])
datal["workex"] = le.fit_transform(datal["workex"])
datal["specialisation"] = le.fit_transform(datal["specialisation"])
datal["status"] = le.fit_transform(datal["status"])

x = datal.iloc[:, :-1]
y = datal["status"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion)

classification_report_output = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report_output)
```

## Output:
![image](https://github.com/user-attachments/assets/010ff115-55c3-4b1e-b9df-23639dfd1c1b)
![image](https://github.com/user-attachments/assets/080749cb-640a-4a84-a968-0a8a47b2ad01)
![image](https://github.com/user-attachments/assets/fcb380fe-3cf4-470a-9c5d-4898b8a5a5dd)
![image](https://github.com/user-attachments/assets/3c40c2ed-fff1-40be-a53e-fe1bbd8fa0fc)
![image](https://github.com/user-attachments/assets/66e0343e-2ddb-44f0-a8dd-1766d9d2f7fb)
![image](https://github.com/user-attachments/assets/27d77193-7e04-4eff-a2ea-cc951a860bc2)
![image](https://github.com/user-attachments/assets/930e26ba-dd36-4937-bb57-4e43b89bd3ba)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
