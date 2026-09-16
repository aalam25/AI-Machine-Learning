# Student Pass/Fail Prediction Using Decision Tree

## Description

This project is my Day 13 machine learning project in my Artificial Intelligence and Machine Learning learning journey.

The goal of this project is to train a Decision Tree Classification model to predict whether a student will pass or fail based on their academic performance.

The project uses student information such as study hours, attendance, assignment scores, midterm scores, and exam scores.

This project builds on the Exploratory Data Analysis and Logistic Regression projects I completed previously.

The main purpose of this project is to understand how a Decision Tree works, how to train and evaluate a classification model, and how changing the tree depth can affect model performance.

---

# Dataset

The project uses a student dataset stored in:

```text
student-data.csv
```

The dataset contains information about student performance.

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

These values are converted into numerical values for machine learning:

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

This project uses a **Decision Tree Classifier** from Scikit-learn.

A Decision Tree is a supervised machine learning algorithm that makes predictions by learning a series of decision rules from the training data.

Conceptually, the model can make decisions based on features such as:

```text
Attendance
    ↓
Study Hours
    ↓
Assignment Score
    ↓
Midterm Score
    ↓
Exam Score
    ↓
Pass / Fail
```

The actual decision rules are learned automatically by the Decision Tree algorithm.

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
Decision Tree
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
Decision Tree Depth Experiment
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

The training data is used to teach the model, while the testing data is used to evaluate how well the model performs on unseen data.

A `random_state` of `42` is used to make the split reproducible.

---

# Model Training

The Decision Tree model is created using:

```python
model = DecisionTreeClassifier(
    random_state=42
)
```

The model is then trained using:

```python
model.fit(X_train, y_train)
```

During training, the Decision Tree learns patterns in the student data that can help it distinguish between passing and failing students.

---

# Predictions

After training, the model makes predictions using:

```python
predictions = model.predict(X_test)
```

The predictions are compared with the actual test results.

The program also displays:

```text
Actual
Predicted
```

for each test student.

This helps show where the model made correct or incorrect predictions.

---

# Model Evaluation

## Accuracy

The project uses accuracy to evaluate the Decision Tree model.

```python
accuracy = accuracy_score(
    y_test,
    predictions
)
```

Accuracy represents the proportion of predictions that were correct.

The general formula is:

```text
Accuracy =
Correct Predictions / Total Predictions
```

A higher accuracy generally means that the model made more correct predictions on the test dataset.

---

# Confusion Matrix

The project also uses a confusion matrix:

```python
cm = confusion_matrix(
    y_test,
    predictions
)
```

A confusion matrix helps show how the model classified each class.

For this project:

```text
0 = Fail
1 = Pass
```

The confusion matrix can contain:

```text
True Negative
False Positive
False Negative
True Positive
```

This provides more detailed information than accuracy alone.

---

# New Student Prediction

The trained model is used to predict the result of a new student.

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

The model then predicts whether the student is likely to:

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

# Decision Tree Depth Experiment

An important part of this project is experimenting with different tree depths.

The project tests:

```text
max_depth = 2
max_depth = 3
max_depth = 5
```

The models are trained and evaluated separately.

Example:

```python
for depth in [2, 3, 5]:

    tree_model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    tree_model.fit(
        X_train,
        y_train
    )

    depth_predictions = tree_model.predict(
        X_test
    )

    depth_accuracy = accuracy_score(
        y_test,
        depth_predictions
    )

    print(
        f"Max Depth: {depth} | "
        f"Accuracy: {depth_accuracy:.2f}"
    )
```

This experiment helps demonstrate how the complexity of a Decision Tree can affect its performance.

---

# What is `max_depth`?

`max_depth` controls the maximum depth of a Decision Tree.

For example:

```text
max_depth = 2
```

creates a relatively simple tree.

A larger value such as:

```text
max_depth = 5
```

allows the tree to create more complex decision rules.

A tree that is too simple may fail to learn important patterns.

A tree that is too complex may learn the training data too closely.

---

# Overfitting

This project introduces the concept of **overfitting**.

Overfitting occurs when a machine learning model learns the training data too closely and does not generalize well to new data.

A simple way to understand it is:

```text
Underfitting
    ↓
Model too simple

Good Fit
    ↓
Learns useful patterns

Overfitting
    ↓
Model too complex
```

The `max_depth` experiment helps demonstrate how controlling model complexity can affect performance.

---

# Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Decision Tree
- Machine Learning

---

# Python Libraries Used

### Pandas

Used for loading and working with the student dataset.

```python
import pandas as pd
```

### NumPy

Used for creating and working with numerical arrays.

```python
import numpy as np
```

### Scikit-learn

Used for machine learning.

The project uses:

```python
train_test_split
DecisionTreeClassifier
accuracy_score
confusion_matrix
```

---

# What I Learned

Through this project, I learned how to build a Decision Tree classification model using Python and Scikit-learn.

I learned how to prepare features and target variables, convert categorical results into numerical values, split a dataset into training and testing sets, train a Decision Tree, and make predictions.

I also learned how to evaluate a classification model using accuracy and a confusion matrix.

One of the most important concepts I learned from this project was model complexity. By experimenting with different `max_depth` values, I learned that changing the complexity of a Decision Tree can affect its predictions and performance.

I also began learning about overfitting and why controlling model complexity is important in machine learning.

---

# Project Structure

```text
Decision Tree
│
├── student_decision_tree.py
├── student-data.csv
└── README.md
```

---

# Project Purpose

This project is part of my ongoing journey toward Artificial Intelligence and Machine Learning.

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

This project builds on those skills by introducing Decision Tree classification.

It also gives me practical experience with training, evaluating, and experimenting with machine learning models.

---

# Future Improvements

In future projects, I plan to:

- Compare Decision Tree with Logistic Regression
- Try other classification algorithms
- Improve model evaluation
- Explore precision, recall, and F1-score
- Visualize Decision Trees
- Investigate feature importance
- Handle larger datasets
- Perform cross-validation
- Tune machine learning model parameters
- Build a complete end-to-end machine learning project

---

# Conclusion

This project helped me understand how a Decision Tree can be used to solve a binary classification problem.

I learned how to prepare student data, train a classification model, evaluate predictions, make predictions for new students, and experiment with model complexity.

The project is an important step in my machine learning learning journey because it moves beyond simply training a model and introduces the importance of evaluating and improving machine learning models.