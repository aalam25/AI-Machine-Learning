# Student Pass/Fail Prediction Using Random Forest

## Description

This project is my Day 14 machine learning project in my Artificial Intelligence and Machine Learning learning journey.

The goal of this project is to train a Random Forest Classification model to predict whether a student will pass or fail based on their academic performance.

The project uses student information such as study hours, attendance, assignment scores, midterm scores, and exam scores.

This project builds on the machine learning concepts I learned in previous projects, including Logistic Regression and Decision Tree Classification.

The main purpose of this project is to understand Random Forest, ensemble learning, model evaluation, feature importance, and model comparison.

---

# Dataset

The project uses a student dataset stored in:

```text
student_data.csv
```

The dataset contains information about student academic performance.

The main features used by the model are:

- Study Hours
- Attendance
- Assignment Score
- Midterm Score
- Exam Score

The target variable is:

- Final Result

The `Final_Result` column contains two possible values:

```text
Pass
Fail
```

These values are converted into numerical values:

```text
Pass → 1
Fail → 0
```

---

# Features

The model uses the following five features:

```text
Study_Hours
Attendance
Assignment_Score
Midterm_Score
Exam_Score
```

These features are stored in `X`.

```python
X = df[
    [
        "Study_Hours",
        "Attendance",
        "Assignment_Score",
        "Midterm_Score",
        "Exam_Score"
    ]
]
```

The target variable is stored in `y`.

```python
y = df["Final_Result"].map({
    "Pass": 1,
    "Fail": 0
})
```

---

# Machine Learning Algorithm

This project uses a **Random Forest Classifier** from Scikit-learn.

Random Forest is an ensemble learning algorithm that combines multiple Decision Trees to make predictions.

Instead of relying on one Decision Tree, Random Forest uses many trees and combines their results.

Conceptually:

```text
                 Random Forest
                       |
        +--------------+--------------+
        |              |              |
        ↓              ↓              ↓
    Tree 1          Tree 2          Tree 3
        |              |              |
        ↓              ↓              ↓
    Prediction     Prediction     Prediction
        |              |              |
        +--------------+--------------+
                       |
                       ↓
                Final Prediction
```

---

# Machine Learning Workflow

The project follows this workflow:

```text
Student Dataset
      ↓
Load Dataset
      ↓
Select Features
      ↓
Create Target
      ↓
Encode Pass/Fail
      ↓
Train/Test Split
      ↓
Random Forest
      ↓
Model Training
      ↓
Predictions
      ↓
Accuracy Evaluation
      ↓
Confusion Matrix
      ↓
New Student Prediction
      ↓
Feature Importance
      ↓
Model Comparison
```

---

# Train/Test Split

The dataset is divided into training and testing sets using:

```python
train_test_split()
```

The project uses:

```python
test_size=0.2
```

This means approximately:

```text
80% → Training Data
20% → Testing Data
```

The training data is used to train the model, while the testing data is used to evaluate the model on data that it did not directly train on.

A `random_state` of `42` is used so that the same train/test split can be reproduced.

---

# Random Forest Model

The Random Forest model is created using:

```python
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

The model uses:

```text
100 Decision Trees
```

The model is trained using:

```python
model.fit(X_train, y_train)
```

---

# What is `n_estimators`?

`n_estimators` controls the number of Decision Trees used by the Random Forest.

For example:

```python
n_estimators=10
```

uses 10 trees.

```python
n_estimators=100
```

uses 100 trees.

```python
n_estimators=200
```

uses 200 trees.

Generally, using more trees can make the model more stable, although it can also require more computation.

---

# Predictions

After training, the model makes predictions using:

```python
predictions = model.predict(X_test)
```

The predictions are compared with the actual test results.

The program also displays the actual and predicted values:

```text
Actual: 1 | Predicted: 1
Actual: 0 | Predicted: 0
```

This helps identify which predictions were correct and which were incorrect.

---

# Model Evaluation

## Accuracy

The project uses accuracy to evaluate the Random Forest model.

```python
accuracy = accuracy_score(
    y_test,
    predictions
)
```

Accuracy represents the proportion of predictions that were correct.

The basic formula is:

```text
Accuracy =
Correct Predictions / Total Predictions
```

A higher accuracy generally means that the model made more correct predictions on the testing data.

---

# Confusion Matrix

The project also uses a confusion matrix:

```python
cm = confusion_matrix(
    y_test,
    predictions
)
```

The target classes are:

```text
0 = Fail
1 = Pass
```

A confusion matrix shows the number of:

- True Negatives
- False Positives
- False Negatives
- True Positives

This provides more detailed information about the model's classification performance than accuracy alone.

---

# New Student Prediction

The trained Random Forest model is used to predict the result of a new student.

Example:

```python
new_student = np.array([[
    6,
    90,
    82,
    85,
    88
]])
```

The values represent:

```text
Study Hours: 6
Attendance: 90
Assignment Score: 82
Midterm Score: 85
Exam Score: 88
```

The model predicts whether the student is likely to:

```text
Pass
```

or:

```text
Fail
```

---

# Multiple Student Predictions

The project also tests the model with multiple new students.

Example:

```python
students = np.array([
    [2, 65, 55, 50, 52],
    [5, 82, 75, 70, 72],
    [8, 95, 92, 90, 94]
])
```

The model predicts the result for each student.

This demonstrates how a trained machine learning model can be used to make predictions for multiple new observations.

---

# Random Forest Experiment

The project experiments with different numbers of Decision Trees.

The following values are tested:

```text
10
50
100
200
```

The models are trained and evaluated separately.

Example:

```python
for trees in [10, 50, 100, 200]:

    forest = RandomForestClassifier(
        n_estimators=trees,
        random_state=42
    )

    forest.fit(
        X_train,
        y_train
    )

    predictions = forest.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        f"Trees: {trees} | "
        f"Accuracy: {accuracy:.2f}"
    )
```

This experiment helps demonstrate how changing the number of trees can affect model performance.

---

# Feature Importance

Random Forest can provide information about the importance of different features.

The project uses:

```python
importance = model.feature_importances_
```

The feature importance values are displayed for:

```text
Study Hours
Attendance
Assignment Score
Midterm Score
Exam Score
```

Example output format:

```text
Study_Hours: 0.150
Attendance: 0.180
Assignment_Score: 0.210
Midterm_Score: 0.220
Exam_Score: 0.240
```

The actual values depend on the dataset and trained model.

Feature importance indicates how useful each feature was for the model's decision-making process.

It does not necessarily mean that a feature causes the final result.

---

# Decision Tree vs Random Forest

One of the main experiments in this project is comparing a Decision Tree with a Random Forest.

The project trains both models using the same training and testing data.

The comparison includes:

```text
Decision Tree Accuracy
Random Forest Accuracy
```

The program then identifies which model achieved the higher accuracy on the test dataset.

Example:

```text
Model Comparison:

Decision Tree Accuracy: 0.XX
Random Forest Accuracy: 0.XX

Best Model:
Random Forest performed better.
```

The actual result depends on the dataset and test split.

The project does not assume that Random Forest will always perform better. Instead, the models are compared using their actual test performance.

---

# Decision Tree vs Random Forest

## Decision Tree

A Decision Tree uses one tree to make predictions.

```text
Dataset
   ↓
Decision Tree
   ↓
Prediction
```

## Random Forest

A Random Forest uses multiple Decision Trees.

```text
Dataset
   ↓
Multiple Decision Trees
   ↓
Combined Predictions
   ↓
Final Prediction
```

Random Forest is an example of **ensemble learning**.

---

# Ensemble Learning

Ensemble learning combines multiple models to produce a prediction.

Random Forest is an ensemble method because it combines many Decision Trees.

The basic idea is:

```text
Multiple Models
       ↓
Combine Their Results
       ↓
Final Prediction
```

This can make the model more robust than relying on a single Decision Tree.

---

# Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Random Forest
- Decision Tree
- Machine Learning

---

# Python Libraries Used

## Pandas

Used for loading and working with the student dataset.

```python
import pandas as pd
```

## NumPy

Used for creating and working with numerical arrays.

```python
import numpy as np
```

## Scikit-learn

Used for machine learning.

The project uses:

```python
train_test_split
RandomForestClassifier
DecisionTreeClassifier
accuracy_score
confusion_matrix
```

---

# What I Learned

Through this project, I learned how Random Forest can be used to solve a binary classification problem.

I learned that Random Forest combines multiple Decision Trees to make predictions.

I also learned how to train a Random Forest model, make predictions, evaluate the model using accuracy and a confusion matrix, and use the trained model to predict the results of new students.

Another important concept I learned was `n_estimators`, which controls the number of Decision Trees used by the Random Forest.

I also learned how to examine feature importance and understand which features the model found useful when making predictions.

Finally, I learned how to compare a Decision Tree and Random Forest using the same dataset and testing data.

---

# Project Structure

```text
Random Forest
│
├── student_random_forest.py
├── student_data.csv
└── README.md
```

---

# Project Purpose

This project is part of my ongoing learning journey toward Artificial Intelligence and Machine Learning.

My previous projects helped me learn:

- Python
- NumPy
- Pandas
- Statistics
- Data Analysis
- Matplotlib
- Linear Regression
- Logistic Regression
- Multiple Features
- Feature Scaling
- Exploratory Data Analysis
- Decision Tree Classification

This project builds on those skills by introducing Random Forest and ensemble learning.

It also gives me practical experience with comparing different machine learning models and analyzing feature importance.

---

# Future Improvements

In future projects, I plan to:

- Compare more machine learning algorithms
- Explore precision, recall, and F1-score
- Perform cross-validation
- Tune model hyperparameters
- Visualize feature importance
- Visualize Decision Trees
- Work with larger datasets
- Improve data preprocessing
- Handle missing values and outliers
- Build an end-to-end machine learning application

---

# Conclusion

This project helped me understand how Random Forest can be used to predict whether students will pass or fail.

I learned how to prepare student data, train a Random Forest classifier, evaluate its predictions, make predictions for new students, experiment with the number of trees, analyze feature importance, and compare Random Forest with a Decision Tree.

This project represents another step forward in my Artificial Intelligence and Machine Learning learning journey and gives me a stronger understanding of classification algorithms and ensemble learning.