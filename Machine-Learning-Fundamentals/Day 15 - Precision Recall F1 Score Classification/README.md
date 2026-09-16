# Student Pass/Fail Prediction - Model Evaluation

## Description

This project is my Day 15 machine learning project in my Artificial Intelligence and Machine Learning learning journey.

The goal of this project is to evaluate a Random Forest Classification model that predicts whether a student will pass or fail based on their academic performance.

The project builds on the Random Forest model I developed in Day 14.

In this project, I focus on evaluating the model using several important classification metrics instead of relying only on accuracy.

The main evaluation metrics used in this project are:

- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report
- Confusion Matrix

The project also examines the distribution of Pass and Fail classes and uses the trained model to predict the result of a new student.

---

# Dataset

The project uses a student dataset stored in:

```text
student_data.csv
```

The dataset contains information about student academic performance.

The features used by the model are:

- Study Hours
- Attendance
- Assignment Score
- Midterm Score
- Exam Score

The target variable is:

- Final Result

The `Final_Result` column contains two classes:

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

The model uses five features:

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

This project uses a **Random Forest Classifier**.

The model is created using:

```python
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

The Random Forest contains 100 Decision Trees.

Random Forest is an ensemble learning algorithm that combines multiple Decision Trees to make predictions.

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
Model Evaluation
      ↓
Classification Metrics
      ↓
Confusion Matrix
      ↓
Class Distribution
      ↓
New Student Prediction
```

---

# Train/Test Split

The dataset is divided into training and testing data using:

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

The training data is used to train the Random Forest model.

The testing data is used to evaluate the model on data that was not used during training.

A `random_state` of `42` is used so that the same train/test split can be reproduced.

---

# Model Training

The Random Forest model is trained using:

```python
model.fit(
    X_train,
    y_train
)
```

After training, the model is used to make predictions:

```python
predictions = model.predict(
    X_test
)
```

---

# Model Evaluation

This project focuses on several different evaluation metrics.

Using multiple metrics provides a better understanding of how the classification model performs.

---

# Accuracy

Accuracy measures the percentage of predictions that were correct.

The project calculates accuracy using:

```python
accuracy = accuracy_score(
    y_test,
    predictions
)
```

The basic formula is:

```text
Accuracy =
Correct Predictions / Total Predictions
```

A higher accuracy generally means that the model made more correct predictions.

However, accuracy alone does not always provide a complete picture of model performance.

---

# Precision

Precision measures how many of the students predicted as the positive class were actually positive.

In this project:

```text
1 = Pass
0 = Fail
```

Precision helps answer:

> Of the students the model predicted would pass, how many actually passed?

The project calculates precision using:

```python
precision = precision_score(
    y_test,
    predictions
)
```

A higher precision means that the model makes fewer incorrect positive predictions.

---

# Recall

Recall measures how many of the actual positive cases were correctly identified by the model.

In this project, the positive class is:

```text
Pass = 1
```

Recall helps answer:

> Of all the students who actually passed, how many did the model correctly identify?

The project calculates recall using:

```python
recall = recall_score(
    y_test,
    predictions
)
```

A higher recall means the model is better at identifying actual positive cases.

---

# F1 Score

F1 Score combines precision and recall into a single metric.

The project calculates it using:

```python
f1 = f1_score(
    y_test,
    predictions
)
```

F1 Score is useful when both precision and recall are important.

It provides a balance between the two metrics.

---

# Classification Report

The project generates a classification report using:

```python
classification_report(
    y_test,
    predictions
)
```

The classification report provides:

- Precision
- Recall
- F1 Score
- Support

for each class.

The two classes are:

```text
0 = Fail
1 = Pass
```

The report provides a more detailed view of the model's classification performance.

---

# Confusion Matrix

The project also generates a confusion matrix:

```python
cm = confusion_matrix(
    y_test,
    predictions
)
```

The confusion matrix shows the number of correct and incorrect predictions for each class.

It can be used to identify:

- True Negatives
- False Positives
- False Negatives
- True Positives

In this project:

```text
0 = Fail
1 = Pass
```

The confusion matrix helps identify what types of mistakes the model is making.

---

# Model Evaluation Summary

The project displays the main metrics together:

```text
Model Evaluation Summary
------------------------
Accuracy:  ...
Precision: ...
Recall:    ...
F1 Score:  ...
```

This makes it easier to compare the different evaluation metrics.

The actual values are generated when the program is executed.

---

# Class Distribution

The project examines how many students belong to each class.

```python
y.value_counts()
```

The project also calculates the percentage distribution:

```python
y.value_counts(
    normalize=True
)
```

This helps determine whether the dataset contains a balanced or imbalanced number of Pass and Fail examples.

For example, if there are significantly more Pass students than Fail students, the dataset may have class imbalance.

Understanding class distribution is important when evaluating classification models.

---

# Actual vs Predicted

The project compares the actual test results with the predictions made by the model.

Example format:

```text
Actual: 1 | Predicted: 1
Actual: 0 | Predicted: 0
```

This helps identify individual correct and incorrect predictions.

---

# New Student Prediction

The trained Random Forest model is also used to predict the result of a new student.

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

# Prediction Probability

The model also calculates the probability of the new student's predicted class.

The project uses:

```python
model.predict_proba(
    new_student
)
```

The pass probability is displayed using:

```python
probability[0][1]
```

Example output:

```text
Pass Probability: 0.92
```

The actual probability will depend on the trained model and dataset.

---

# Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Random Forest
- Classification
- Machine Learning

---

# Python Libraries Used

## Pandas

Pandas is used to load and work with the student dataset.

```python
import pandas as pd
```

## NumPy

NumPy is used for numerical arrays and preparing the new student data.

```python
import numpy as np
```

## Scikit-learn

Scikit-learn is used for machine learning and model evaluation.

The project uses:

```python
train_test_split
RandomForestClassifier
accuracy_score
precision_score
recall_score
f1_score
classification_report
confusion_matrix
```

---

# What I Learned

Through this project, I learned that evaluating a machine learning model involves more than simply checking its accuracy.

I learned how to calculate and interpret:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

I learned that precision focuses on the correctness of positive predictions, while recall focuses on identifying actual positive cases.

I also learned that the F1 Score provides a balance between precision and recall.

Another important concept I learned was class distribution and how an imbalanced dataset can affect the interpretation of model performance.

I also learned how to examine actual versus predicted results and how to use a trained Random Forest model to make predictions for new students.

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
- Random Forest Classification

This project builds on those skills by focusing on professional model evaluation.

It helps me understand how to determine whether a classification model is performing well and how different evaluation metrics provide different information about the model.

---

# Future Improvements

In future projects, I plan to:

- Compare additional machine learning algorithms
- Perform cross-validation
- Tune model hyperparameters
- Handle class imbalance
- Explore ROC curves
- Calculate ROC-AUC
- Visualize confusion matrices
- Visualize model performance
- Work with larger datasets
- Improve data preprocessing
- Build an end-to-end machine learning application

---

# Conclusion

This project helped me understand how to properly evaluate a Random Forest classification model.

Instead of relying only on accuracy, I learned how to use precision, recall, F1 Score, classification reports, and confusion matrices to better understand model performance.

I also learned how to examine class distribution, compare actual and predicted values, and make predictions for new students.

This project represents another step in my Artificial Intelligence and Machine Learning learning journey and gives me a stronger foundation in classification model evaluation.