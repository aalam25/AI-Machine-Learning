# Student Performance Prediction with Feature Scaling

## Description

This project is my fourth machine learning project and builds on the classification and multiple-feature concepts I learned in my previous projects.

The goal of this project is to predict whether a student will pass or fail using multiple student performance features.

The model uses:

- Study Hours
- Attendance
- Assignment Score
- Exam Score

A Logistic Regression model is used to predict the student's result:

- `0` = Fail
- `1` = Pass

This project introduces an important machine learning concept called **Feature Scaling**.

Feature scaling helps transform numerical features to a more comparable scale before training a machine learning model.

---

## Dataset

The project uses a small sample dataset containing student performance information.

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

The target variable represents the student's result:

```text
0 → Fail
1 → Pass
```

---

## Features

The model uses four features.

### Study Hours

The number of hours the student studies.

### Attendance

The student's attendance percentage.

### Assignment Score

The student's assignment score.

### Exam Score

The student's exam score.

These features are combined into a feature matrix and used as inputs for the machine learning model.

---

## Why Feature Scaling?

The features in this dataset have different numerical ranges.

For example:

```text
Study Hours       → 2–8
Attendance        → 70–98
Assignment Score  → 60–90
Exam Score        → 55–95
```

Because the features have different scales, feature scaling can transform them into a more comparable numerical range.

This project uses `StandardScaler` from Scikit-learn.

---

## StandardScaler

The project uses:

```python
from sklearn.preprocessing import StandardScaler
```

A scaler is created using:

```python
scaler = StandardScaler()
```

The original features are then scaled using:

```python
X_scaled = scaler.fit_transform(X)
```

`StandardScaler` standardizes the features based on their mean and standard deviation.

After scaling, the features are generally centered around zero with a standard deviation close to one.

---

## Feature Matrix

The multiple features are combined using NumPy:

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

The structure of the model is:

```text
Study Hours
      |
Attendance
      |
Assignment Score
      |
Exam Score
      |
      ↓
Feature Scaling
      ↓
Logistic Regression
      ↓
Pass / Fail
```

---

## Machine Learning Workflow

The overall workflow of this project is:

```text
Student Dataset
       ↓
Feature Matrix
       ↓
Feature Scaling
       ↓
Train/Test Split
       ↓
Logistic Regression
       ↓
Model Training
       ↓
Predictions
       ↓
Accuracy Evaluation
       ↓
New Student Prediction
```

---

## Train/Test Split

The scaled dataset is divided into training and testing data.

The project uses:

```python
train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)
```

The model uses:

- 80% of the data for training
- 20% of the data for testing

The training data is used to train the model, while the testing data is used to evaluate its predictions.

---

## Logistic Regression

This project uses Logistic Regression for binary classification.

The model is created using:

```python
model = LogisticRegression()
```

The model is trained using:

```python
model.fit(X_train, y_train)
```

After training, the model can predict whether a student will pass or fail.

---

## Model Predictions

The model makes predictions using:

```python
predictions = model.predict(X_test)
```

The predictions represent:

```text
0 = Fail
1 = Pass
```

The model's predictions are then compared with the actual test results.

---

## Model Accuracy

The project uses accuracy to evaluate the model.

Accuracy is calculated using:

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(
    y_test,
    predictions
)
```

Accuracy represents the proportion of correct predictions made by the model.

A higher accuracy generally means the model made more correct predictions on the test data.

---

## Predicting a New Student

The trained model can be used to predict the result of a new student.

Example:

```text
Study Hours: 6
Attendance: 90%
Assignment Score: 82
Exam Score: 85
```

The new student's data is first converted into a NumPy array:

```python
new_student = np.array([[
    6,
    90,
    82,
    85
]])
```

The new student must then be scaled using the same scaler that was fitted to the original dataset:

```python
new_student_scaled = scaler.transform(new_student)
```

The model can then make a prediction:

```python
prediction = model.predict(
    new_student_scaled
)
```

---

## Prediction Probability

The project also calculates the probability of the student passing.

This is done using:

```python
probability = model.predict_proba(
    new_student_scaled
)
```

The Pass probability is displayed using:

```python
probability[0][1]
```

For example:

```text
Pass Probability: 0.85
```

would represent an estimated 85% probability of the student being classified as Pass.

The actual probability is generated by the trained model when the program runs.

---

## Important Difference Between `fit_transform()` and `transform()`

One of the important concepts learned in this project is the difference between:

```python
fit_transform()
```

and:

```python
transform()
```

### `fit_transform()`

This is used on the original dataset.

```python
X_scaled = scaler.fit_transform(X)
```

The scaler learns the required scaling information from the dataset and then transforms the data.

### `transform()`

This is used for new data.

```python
new_student_scaled = scaler.transform(new_student)
```

The same scaling information learned from the original dataset is applied to the new student.

The scaler should not be fitted again on the new student's data.

---

## Reusable Prediction Function

The project includes a reusable function called:

```python
predict_student()
```

The function accepts:

- Study Hours
- Attendance
- Assignment Score
- Exam Score

It then:

1. Creates the student's feature array.
2. Scales the student's data.
3. Makes a Pass/Fail prediction.
4. Calculates the Pass probability.
5. Displays the student's information and prediction.

Example:

```python
predict_student(6, 90, 82, 85)
```

Additional examples:

```python
predict_student(2, 65, 55, 50)

predict_student(8, 98, 95, 95)
```

This makes it possible to test the trained model with different student profiles.

---

## Technologies Used

- Python
- NumPy
- Scikit-learn
- Logistic Regression
- StandardScaler

---

## Libraries Used

### NumPy

NumPy is used to:

- Create numerical arrays
- Store student data
- Combine multiple features
- Prepare data for machine learning

### Scikit-learn

Scikit-learn is used for:

- Feature scaling
- Train/test splitting
- Logistic Regression
- Model predictions
- Prediction probabilities
- Accuracy evaluation

---

## What I Learned

Through this project, I learned how feature scaling can be used in a machine learning workflow.

I learned that different features can have very different numerical ranges. For example, study hours may range from 2 to 8, while attendance may range from 70 to 98.

I learned how to use `StandardScaler` to transform these features into a more comparable scale.

I also learned the difference between `fit_transform()` and `transform()`.

Another important lesson was that new data must be transformed using the same scaler that was fitted to the original dataset.

I also continued practicing Logistic Regression, multiple features, train/test splitting, predictions, accuracy evaluation, and prediction probabilities.

---

## Model Limitations

This project uses a very small sample dataset containing only 10 students.

Because the dataset is very small, the model's accuracy should not be considered a reliable measurement of real-world student performance.

The purpose of this project is to learn the machine learning workflow and understand feature scaling rather than create a production-ready prediction system.

A real-world application would require a much larger and more representative dataset.

---

## Future Improvements

Possible improvements include:

- Using a larger real-world dataset
- Adding more student features
- Comparing different scaling methods
- Comparing different machine learning algorithms
- Using cross-validation
- Performing hyperparameter tuning
- Adding data visualization
- Improving model evaluation
- Building a simple user interface
- Deploying the model as a web application

---

## Project Purpose

This project is part of my learning journey toward Artificial Intelligence and Machine Learning.

It builds on the Python, NumPy, Pandas, data analysis, visualization, statistics, Linear Regression, Logistic Regression, and multiple-feature classification skills I developed in my previous projects.

This project helped me understand an important machine learning preprocessing technique: **feature scaling**.

It also helped me understand how training data and new prediction data must be processed consistently before being given to a machine learning model.

---

## Conclusion

This project demonstrates a basic machine learning classification workflow using multiple features and feature scaling.

A Logistic Regression model is trained using study hours, attendance, assignment scores, and exam scores to predict whether a student will pass or fail.

The project also demonstrates how `StandardScaler` can be used to standardize numerical features and how the same scaler should be applied to new data before making predictions.

This project represents another step in my journey toward learning Artificial Intelligence and Machine Learning.