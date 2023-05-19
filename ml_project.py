# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Load your chosen dataset into a pandas DataFrame
data = pd.read_csv('Training.csv')  # Replace 'your_dataset.csv' with the actual path to your dataset file

# Separate the features and the target variable
X = data.drop('target', axis=1)
y = data['target']
# Step 3: Running ML Algorithms on the Dataset
# Preprocess the data (assuming you have already handled missing values and performed feature engineering)
# Split the dataset into training and testing subsets
X = data.drop(target_class, axis=1)
y = data[target_class]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose and train an ML algorithm
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test dataset
y_pred = model.predict(X_test)

# Step 4: Calculating Confusion Matrix and Evaluation Metrics
# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1-score: ", f1)
