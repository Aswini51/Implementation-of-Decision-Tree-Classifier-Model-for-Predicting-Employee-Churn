# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load Data – Import the employee dataset with relevant features and churn labels.

2. Preprocess Data – Handle missing values, encode categorical features, and split into train/test sets.

3. Initialize Model – Create a DecisionTreeClassifier with desired parameters.

4. Train Model – Fit the model on the training data.

5. Evaluate Model – Predict on test data and check accuracy, precision, recall, etc.

6. Visualize & Interpret – Visualize the tree and identify key features influencing churn. 

## Program and Output:
/*

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

Developed by: G ASWINI

RegisterNumber:  212224040037

*/

```
import pandas as pd
data = pd.read_csv("C:\\Users\\admin\\OneDrive\\Desktop\\Folders\\ML\\DATASET-20250226\\Employee.csv")
data.head()
```
![Screenshot 2025-04-24 092228](https://github.com/user-attachments/assets/d984b2fb-9820-45a7-b3ef-b550c903a6e6)

```
data.info()
data.isnull().sum()
data['left'].value_counts()
```
![Screenshot 2025-04-24 092241](https://github.com/user-attachments/assets/76506a59-7213-43b1-b66e-79eac79e56d5)

```
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['salary'] = le.fit_transform(data['salary'])
data.head()
```
![Screenshot 2025-04-24 092301](https://github.com/user-attachments/assets/f9549631-2268-4346-b72a-16cd323c3b82)

```
x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()
```
![Screenshot 2025-04-24 092313](https://github.com/user-attachments/assets/1dbe360b-ef38-4ee6-a05d-ea9155785591)

```
y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state =100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy
```
![Screenshot 2025-04-24 092320](https://github.com/user-attachments/assets/acf94cbf-afd2-4b7a-aca3-a441556a0cfb)

```
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
![Screenshot 2025-04-24 092327](https://github.com/user-attachments/assets/864f75e5-8279-4e5e-b37e-e646bcaed25a)

```
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree # Import the plot_tree function

plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```
![Screenshot 2025-04-24 092343](https://github.com/user-attachments/assets/2264fa2a-c89c-432f-b772-2a5b2aa8c3a9)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
