# Student Performance Prediction Using Multiple Features

## Description

This project is my third machine learning project and builds on the classification concepts I learned in my previous Logistic Regression project.

The goal of this project is to predict whether a student will pass or fail using multiple features instead of relying on only one feature.

The model uses:

- Study Hours
- Attendance
- Assignment Score
- Exam Score

A Logistic Regression model is trained to predict the student's final result:

- `0` = Fail
- `1` = Pass

This project demonstrates how multiple features can be combined and used as inputs for a machine learning classification model.

---

## Dataset

The project uses a small sample dataset containing information about student performance.

| Study Hours | Attendance | Assignment Score | Exam Score | Result |
|------------:|-----------:|-----------------:|-----------:|--------|
| 2 | 70 | 60 | 55 | Fail |
| 3 | 75 | 65 | 60 | Fail |
| 4 | 80 | 70 | 68 | Pass |
| 5 | 85 | 75 | 72 | Pass |
| 6 | 90 | 80 | 80 | Pass |
| 7 | 95 | 85 | 88 | Pass |
| 8 | 98 | 90 | 95 | Pass |
| 3 | 78 | 68 | 62 | Fail |
| 5 | 88 | 78 | 75 | Pass |
| 7 | 92 | 88 | 90 | Pass |

### Target Variable

The target variable represents whether the student passed or failed:

```text
0 → Fail
1 → Pass
```

---

## Features

The model uses four features:

### 1. Study Hours

The number of hours a student studies.

### 2. Attendance

The student's attendance percentage.

### 3. Assignment Score

The student's assignment score.

### 4. Exam Score

The student's exam score.

These four features are combined to create the machine learning feature matrix.

---

## Machine Learning Approach

This project uses **Logistic Regression** for binary classification.

The model learns a relationship between multiple student performance features and the student's Pass/Fail result.

### Features

```text
Study Hours
Attendance
Assignment Score
Exam Score
```

### Target

```text
Pass / Fail
```

---

## Feature Matrix

The project uses NumPy's `column_stack()` function to combine the different features.

```python
X = np.column_stack((
    study_hours,
    attendance,
    assignment_scores,
    exam_scores
))
```

The target variable is:

```python
y = passed
```

This creates the following machine learning structure:

```text
Study Hours ────────┐
Attendance ─────────┤
Assignment Score ───┤
Exam Score ─────────┤
                    ↓
            Logistic Regression
                    ↓
                Pass / Fail
```

---

## Machine Learning Workflow

```text
Student Dataset
       ↓
Multiple Features
       ↓
Feature Matrix (X)
       ↓
Target Variable (y)
       ↓
Train/Test Split
       ↓
Logistic Regression
       ↓
Model Training
       ↓
Predictions
       ↓
Model Evaluation
       ↓
New Student Prediction
```

---

## Train/Test Split

The dataset is divided into training and testing data.

The project uses:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

The model uses 80% of the data for training and 20% for testing.

The training data is used to teach the model, while the testing data is used to evaluate how well the model performs on unseen data.

---

## Model Training

The project creates a Logistic Regression model:

```python
model = LogisticRegression()
```

The model is trained using:

```python
model.fit(X_train, y_train)
```

After training, the model can make predictions for new students.

---

## Model Predictions

The model predicts whether students in the test dataset will pass or fail.

The prediction is generated using:

```python
predictions = model.predict(X_test)
```

The program also compares the actual results with the predicted results.

Example:

```text
Actual: 1 | Predicted: 1
Actual: 0 | Predicted: 0
```

Where:

```text
0 = Fail
1 = Pass
```

---

## Model Evaluation

The project evaluates the model using:

- Accuracy
- Confusion Matrix

### Accuracy

Accuracy measures how many predictions the model classified correctly.

The project calculates accuracy using:

```python
accuracy_score(
    y_test,
    predictions
)
```

A higher accuracy generally means that the model made more correct predictions on the test data.

---

## Confusion Matrix

The project also uses a confusion matrix:

```python
confusion_matrix(
    y_test,
    predictions
)
```

A confusion matrix helps show how many students were correctly or incorrectly classified as:

- Fail
- Pass

It provides more detail about the model's classification performance than accuracy alone.

---

## New Student Prediction

The trained model is used to predict the result of a new student.

Example student:

```text
Study Hours: 6
Attendance: 90%
Assignment Score: 82
Exam Score: 85
```

The model predicts whether this student is likely to pass or fail.

The program also calculates the probability that the student will pass.

---

## Prediction Probability

The project uses:

```python
model.predict_proba()
```

to calculate the probability of each class.

For example:

```text
Fail Probability → 0.15
Pass Probability → 0.85
```

This would mean the model estimates an 85% probability that the student will pass.

The actual probability is generated by the trained model when the program runs.

---

## Reusable Prediction Function

A reusable function called `predict_student()` was created.

```python
def predict_student(
    study_hours,
    attendance,
    assignment_score,
    exam_score
):
```

This function allows predictions to be made for different students without rewriting the prediction code.

For example:

```python
predict_student(6, 90, 82, 85)

predict_student(2, 65, 55, 50)

predict_student(8, 98, 95, 95)
```

This demonstrates how a trained machine learning model can be reused to make predictions for different inputs.

---

## Technologies Used

- Python
- NumPy
- Scikit-learn
- Logistic Regression

---

## Libraries Used

### NumPy

NumPy is used to create and organize the numerical data and combine multiple features into a feature matrix.

### Scikit-learn

Scikit-learn is used for:

- Train/test splitting
- Logistic Regression
- Accuracy calculation
- Confusion matrix
- Prediction probabilities

---

## What I Learned

Through this project, I learned how to use multiple features in a machine learning model.

I learned that a machine learning model does not have to rely on only one input variable. Multiple features can be combined to provide more information to the model.

I learned how to use `np.column_stack()` to create a feature matrix containing multiple input variables.

I also practiced splitting data into training and testing sets, training a Logistic Regression model, making predictions, calculating prediction probabilities, and evaluating a classification model using accuracy and a confusion matrix.

Another important concept I learned was how to create a reusable prediction function that can accept different student information and use the trained model to make predictions.

---

## Day 8 vs Day 9

In my previous project, the model used only one feature:

```text
Exam Score
     ↓
Logistic Regression
     ↓
Pass / Fail
```

In this project, the model uses multiple features:

```text
Study Hours
Attendance
Assignment Score
Exam Score
        ↓
Logistic Regression
        ↓
Pass / Fail
```

This helped me understand how machine learning models can use multiple pieces of information to make a prediction.

---

## Project Limitations

This project uses a very small sample dataset containing only 10 students.

Because the dataset is small, the model's performance should not be considered representative of a real-world student population.

The purpose of this project is to understand the machine learning workflow and learn how multiple features can be used in a classification problem.

A larger and more realistic dataset would be required for a reliable real-world prediction system.

---

## Project Purpose

This project is part of my learning journey toward Artificial Intelligence and Machine Learning.

It builds on the Python, NumPy, Pandas, data analysis, visualization, statistics, Linear Regression, and Logistic Regression skills I developed in previous projects.

This project helped me move from a simple one-feature classification model to a model that uses multiple features.

My next goal is to continue learning machine learning concepts and eventually build projects using real-world datasets.

---

## Future Improvements

Possible improvements for this project include:

- Using a larger real-world dataset
- Adding more student features
- Comparing different machine learning algorithms
- Improving model evaluation
- Visualizing the dataset
- Performing feature scaling
- Performing cross-validation
- Tuning model parameters
- Comparing Logistic Regression with other classification algorithms

---

## Project Structure

```text
Day 9 - Multiple Features
│
├── student_performance.py
└── README.md
```

---

## Conclusion

This project demonstrates a basic machine learning classification workflow using multiple features.

The Logistic Regression model uses study hours, attendance, assignment scores, and exam scores to predict whether a student will pass or fail.

This project helped strengthen my understanding of feature matrices, classification, model training, prediction, evaluation, and reusable machine learning functions.