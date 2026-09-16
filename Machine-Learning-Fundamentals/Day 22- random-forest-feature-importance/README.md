# Random Forest Feature Importance Visualization

## Overview

This project explores **feature importance in a Random Forest classifier** using a student performance dataset.

The goal is to understand which student-related features contribute the most to predicting whether a student will **Pass or Fail**.

This project is part of my practical journey in learning **Machine Learning and Artificial Intelligence**.

---

## Project Objective

The main objectives of this project are to:

- Train a Random Forest classification model.
- Calculate feature importance.
- Rank features from most important to least important.
- Identify the feature that contributes the most to the model's predictions.
- Visualize feature importance using a bar chart.
- Understand how machine learning models can help explain which inputs are most useful for prediction.

---

## Dataset

The project uses a student performance dataset containing **50 students** and **7 columns**.

The dataset includes:

- Student ID
- Study Hours
- Attendance
- Assignment Score
- Midterm Score
- Exam Score
- Final Result

For this project, four features were selected for the Random Forest model.

---

## Features Used

The following features were used:

1. `Study_Hours`
2. `Attendance`
3. `Assignment_Score`
4. `Exam_Score`

The `Midterm_Score` column was not used in this project.

---

## Target Variable

The target variable is:

`Final_Result`

The categorical values were converted into numerical values:

- `Pass` → `1`
- `Fail` → `0`

This allows the Random Forest classifier to work with the target variable.

---

## Machine Learning Model

I used a **Random Forest Classifier**.

The model was created with:

- `n_estimators = 100`
- `random_state = 42`

Random Forest is an ensemble machine learning algorithm that combines multiple decision trees to make predictions.

---

## Train-Test Split

The dataset was divided into training and testing sets.

- Training data: **40 rows**
- Testing data: **10 rows**
- Test size: **20%**
- Random state: **42**

The training data was used to train the Random Forest model.

---

## Feature Importance

After training the model, I used the Random Forest `feature_importances_` attribute to determine the relative importance of each feature.

The results were:

| Feature | Importance |
|---|---:|
| Exam_Score | 0.3555 |
| Attendance | 0.2808 |
| Assignment_Score | 0.2787 |
| Study_Hours | 0.0849 |

The values represent the relative contribution of each feature to the Random Forest model's decision-making.

---

## Feature Importance Ranking

The features were ranked from most important to least important:

### 1. Exam_Score

Importance: **0.3555**

Exam Score was the most important feature in this Random Forest model.

### 2. Attendance

Importance: **0.2808**

Attendance was the second most important feature.

### 3. Assignment_Score

Importance: **0.2787**

Assignment Score was very close to Attendance in importance.

### 4. Study_Hours

Importance: **0.0849**

Study Hours had the lowest feature importance among the four selected features.

---

## Most Important Feature

The most important feature identified by the model was:

**Exam_Score**

Its feature importance value was:

**0.3555**

This means that, among the four features used in this model and dataset, Exam Score contributed the most to the Random Forest's predictions.

It does not mean that Study Hours are unimportant in real life. Feature importance describes how useful each feature was to this particular model on this particular dataset.

---

## Visualization

A bar chart was created using Matplotlib to visualize the feature importance values.

The chart is saved as:

`feature_importance.png`

The visualization makes it easier to compare the contribution of each feature.

---

## Project Structure

    random-forest-feature-importance/
    │
    ├── feature_importance.py
    ├── student_data.csv
    ├── feature_importance.png
    └── README.md

---

## Python Libraries Used

### Pandas

Used for:

- Loading the dataset
- Selecting features
- Creating the feature importance DataFrame
- Sorting the feature importance values

### Scikit-learn

Used for:

- Splitting the dataset
- Creating the Random Forest model
- Calculating feature importance

### Matplotlib

Used for:

- Creating the feature importance bar chart
- Saving the visualization as a PNG file

---

## How to Run the Project

### 1. Clone the repository

Clone this repository to your computer.

### 2. Open the project folder

Navigate to:

    random-forest-feature-importance

### 3. Make sure the dataset is available

Make sure `student_data.csv` is in the same folder as `feature_importance.py`.

### 4. Run the Python program

Use:

    python feature_importance.py

The program will display:

- The dataset
- Selected features
- Target values
- Training and testing data shapes
- Feature importance values
- Feature importance ranking
- The most important feature

It will also create:

    feature_importance.png

---

## Example Results

    Training Data Shape:
    (40, 4)

    Testing Data Shape:
    (10, 4)

    Random Forest Model Trained Successfully.

    Feature Importance:
    [0.08493813 0.28077093 0.27874996 0.35554098]

    Feature Importance Ranking:

    Exam_Score          0.355541
    Attendance          0.280771
    Assignment_Score    0.278750
    Study_Hours         0.084938

    Most Important Feature:
    Exam_Score

    Importance Value:
    0.3555

---

## What I Learned

Through this project, I learned:

- How Random Forest models calculate feature importance.
- How to use the `feature_importances_` attribute.
- How to organize feature importance values using Pandas.
- How to sort features based on their importance.
- How to identify the most important feature.
- How to create a feature importance visualization using Matplotlib.
- How model interpretation can help us understand machine learning predictions.
- Why feature importance results depend on the dataset and the model being used.

---

## Key Takeaway

Feature importance is useful because it provides a simple way to understand which features contribute most to a machine learning model's predictions.

In this project, **Exam_Score** was the most important feature, followed by **Attendance**, **Assignment_Score**, and **Study_Hours**.

This project helped me move beyond simply training a machine learning model and start understanding **why a model may rely on certain features more than others**.

---

## Future Improvements

Possible improvements include:

- Compare Random Forest feature importance with Decision Tree feature importance.
- Use permutation importance for another interpretation method.
- Experiment with different numbers of trees.
- Compare feature importance across different Random Forest configurations.
- Use a larger student dataset.
- Explore additional machine learning models.
- Create more advanced model interpretation visualizations.

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Random Forest
- Machine Learning
- Git
- GitHub

---

## Project Context

This project is part of my hands-on Machine Learning learning path.

So far, I have worked with:

- Python
- NumPy
- Pandas
- Data Analysis
- Data Visualization
- Decision Trees
- Random Forest
- Cross-Validation
- GridSearchCV
- Model Evaluation
- Confusion Matrix
- Feature Importance

This project continues that progression by focusing on **interpreting machine learning models**.

---

## Author

**Abone Alam**

Computer Science Student | Aspiring AI/ML Professional

GitHub: `aalam25`