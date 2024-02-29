# EXP 2: Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
![image](https://github.com/lathishlathish/nn-classification/assets/120359170/424d16a6-f4b0-44c3-9c04-77227531bb81)


## DESIGN STEPS

### STEP 1:
Import the necessary packages and modules
### STEP 2:
Load and read the dataset
### STEP 3:
Pre-process and clean the dataset
### STEP 4:
Encode categorical value into numerical values using ordinal/label/one hot encoders
### STEP 5:
Visualize the data in different plots using seaborn
### STEP 6:
Normalize the values and split them for x and y
### STEP 7:
Build the deep learning model with appropriate layers and depth
### STEP 8:
Analyze the model using different metrics
### STEP 9:
Plot a graph for Training Loss, Validation Loss vs Iteration and for Accuracy, Validation Accuracy vs Iteration
### STEP 10:
Save the model using pickle
### STEP 11:
Using the DL model predict some random inputs
## PROGRAM

### Name: LATHISH KANNA.M
### Register Number: 212222230073

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pylab as plt

customer_df = pd.read_csv('customers.csv')
customer_df.columns
customer_df.dtypes
customer_df.shape
customer_df.isnull().sum()
customer_df_cleaned = customer_df.dropna(axis = 0)
customer_df_cleaned.isnull().sum()
customer_df_cleaned.shape
customer_df_cleaned.dtypes
customer_df_cleaned['Gender'].unique()
customer_df_cleaned['Ever_Married'].unique()
customer_df_cleaned['Graduated'].unique()
customer_df_cleaned['Profession'].unique()
customer_df_cleaned['Spending_Score'].unique()
customer_df_cleaned['Var_1'].unique()
customer_df_cleaned['Segmentation'].unique()

categories_list = [['Male', 'Female'],
                   ['No', 'Yes'],
                   ['No', 'Yes'],
                   ['Healthcare', 'Engineer', 'Lawyer', 'Artist', 'Doctor', 'Homemaker', 'Entertainment', 'Marketing', 'Executive'],
                   ['Low', 'Average', 'High']]
enc = OrdinalEncoder(categories = categories_list)

customers_1 = customer_df_cleaned.copy()
customers_1[['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score']] = enc.fit_transform(customers_1[['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score']])
customers_1.dtypes
le = LabelEncoder()
customers_1['Segmentation'] = le.fit_transform(customers_1['Segmentation'])
customers_1.dtypes
customers_1 = customers_1.drop('ID',axis=1)
customers_1 = customers_1.drop('Var_1',axis=1)
customers_1.dtypes

corr = customers_1.corr()
sns.heatmap(corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cmap="BuPu",
        annot= True)
sns.pairplot(customers_1)
sns.distplot(customers_1['Age'])
plt.figure(figsize=(10,6))
sns.countplot(customers_1['Family_Size'])
plt.figure(figsize=(10,6))
sns.boxplot(x='Family_Size',y='Age',data=customers_1)
plt.figure(figsize=(10,6))
sns.scatterplot(x='Family_Size',y='Spending_Score',data=customers_1)
plt.figure(figsize=(10,6))
sns.scatterplot(x='Family_Size',y='Age',data=customers_1)

customers_1.describe()
customers_1['Segmentation'].unique()
X=customers_1[['Gender','Ever_Married','Age','Graduated','Profession','Work_Experience','Spending_Score','Family_Size']]
y1 = customers_1[['Segmentation']].values
one_hot_enc = OneHotEncoder()
one_hot_enc.fit(y1)
y1.shape
y = one_hot_enc.transform(y1).toarray()
y.shape
y1[0]
y[0]
X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 50)
X_train.index
X_train.shape
scaler_age = MinMaxScaler()
y1
scaler_age.fit(X_train.iloc[:, 2].values.reshape(-1,1))
X_train_scaled = np.copy(X_train)
X_test_scaled = np.copy(X_test)
X_train_scaled[:,2] = scaler_age.transform(X_train.iloc[:,2].values.reshape(-1,1)).reshape(-1)
X_test_scaled[:,2] = scaler_age.transform(X_test.iloc[:,2].values.reshape(-1,1)).reshape(-1)

from tensorflow.keras.callbacks import EarlyStopping

ai_brain = Sequential()
X_train.shape[1]
X_train_scaled.shape
ai_brain.add(Dense(5, activation = "relu", input_shape = (X_train.shape[1],)))
ai_brain.add(Dense(10, activation = "relu"))
ai_brain.add(Dense(10, activation = "relu"))
ai_brain.add(Dense(4, activation = "softmax"))
ai_brain.summary()
ai_brain.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=2)
ai_brain.fit(x = X_train_scaled, y = y_train, epochs = 2000, batch_size = 32, validation_data = (X_test_scaled, y_test),)
metrics = pd.DataFrame(ai_brain.history.history)
metrics.head()
metrics[['loss','val_loss']].plot()
x_test_predictions = np.argmax(ai_brain.predict(X_test_scaled), axis=1)
x_test_predictions.shape
y_test_truevalue = np.argmax(y_test,axis=1)
y_test_truevalue.shape
print(confusion_matrix(y_test_truevalue,x_test_predictions))
print(classification_report(y_test_truevalue,x_test_predictions))
ai_brain.save('customer_classification_model.h5')
with open('customer_data.pickle', 'wb') as fh:
   pickle.dump([X_train_scaled,y_train,X_test_scaled,y_test,customers_1,customer_df_cleaned,scaler_age,enc,one_hot_enc,le], fh)
ai_brain = load_model('customer_classification_model.h5')
with open('customer_data.pickle', 'rb') as fh:
   [X_train_scaled,y_train,X_test_scaled,y_test,customers_1,customer_df_cleaned,scaler_age,enc,one_hot_enc,le]=pickle.load(fh)

x_single_prediction = np.argmax(ai_brain.predict(X_test_scaled[1:2,:]), axis=1)
print(x_single_prediction)
print(le.inverse_transform(x_single_prediction))
```

## Dataset Information
![image](https://github.com/lathishlathish/nn-classification/assets/120359170/73d2838c-b4ae-4b8c-ba52-a85f13311142)


## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
![image](https://github.com/lathishlathish/nn-classification/assets/120359170/6d4910da-e034-4e56-982e-911c27c13dc6)

### Classification Report
![image](https://github.com/lathishlathish/nn-classification/assets/120359170/0d1d72dc-d9f0-45d3-8224-261698d877d2)

### Confusion Matrix
![image](https://github.com/lathishlathish/nn-classification/assets/120359170/975220fc-708c-4447-be65-c19a6b414b6d)

### New Sample Data Prediction
![image](https://github.com/lathishlathish/nn-classification/assets/120359170/32929dbc-d0d6-4923-9055-bb41a8b10ce5)

## RESULT
A neural network classification model is developed for the given dataset.
