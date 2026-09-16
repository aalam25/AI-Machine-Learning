# Day 19 - Cross-Validation and Model Comparison

## Project Overview

This project is part of my machine learning learning journey.

In Day 19, I learned about **Cross-Validation** and how it can provide a more reliable evaluation of a machine learning model.

I used a **Random Forest Classifier** to predict whether a student will Pass or Fail based on student performance data.

I also completed an optional challenge where I compared a **Decision Tree** with a **Random Forest** using 5-fold cross-validation.

---

## Objective

The main objectives of this project are:

- Understand cross-validation
- Learn about K-Fold Cross-Validation
- Use `cross_val_score()` from Scikit-learn
- Calculate average cross-validation accuracy
- Find the highest and lowest cross-validation accuracy
- Compare different numbers of folds
- Compare Decision Tree and Random Forest models
- Understand why cross-validation can provide a more reliable model evaluation

---

## Dataset

The project uses a student dataset called:

`student_data.csv`

The dataset contains student academic information.

### Features Used

- `Study_Hours` - Number of hours the student studies
- `Attendance` - Student attendance percentage
- `Assignment_Score` - Assignment score
- `Exam_Score` - Exam score

### Target

The target column is:

`Final_Result`

It contains two possible values:

- `Pass`
- `Fail`

For machine learning, these values are converted into numbers:

- `Pass` = 1
- `Fail` = 0

---

## Machine Learning Models

### Random Forest

Random Forest is an ensemble machine learning algorithm that combines multiple Decision Trees.

In this project, I used:

`n_estimators=100`

This means the Random Forest contains 100 decision trees.

### Decision Tree

A Decision Tree is a machine learning model that makes decisions using a tree-like structure.

I used it in the optional challenge to compare its performance with Random Forest.

---

## Train-Test Split

First, I divided the dataset into training and testing data.

I used:

- 80% training data
- 20% testing data

The code uses:

`train_test_split()`

with:

`random_state=42`

This gives a consistent split each time the program runs.

---

## Cross-Validation

Cross-validation is a technique used to evaluate a machine learning model using multiple train/test splits.

Instead of depending on only one train/test split, cross-validation divides the dataset into multiple folds.

For example, in **5-fold cross-validation**:

1. The dataset is divided into 5 parts.
2. Four parts are used for training.
3. One part is used for validation.
4. This process is repeated 5 times.
5. Each part gets a chance to be used for validation.
6. The results are combined to calculate the average accuracy.

---

## `cross_val_score()`

Scikit-learn provides the `cross_val_score()` function for cross-validation.

Example:

`cross_val_score(model, X, y, cv=5, scoring="accuracy")`

Here:

- `model` is the machine learning model
- `X` contains the features
- `y` contains the target
- `cv=5` means 5-fold cross-validation
- `scoring="accuracy"` means accuracy is used for evaluation

---

## Cross-Validation Accuracy

The project calculates three important values for 5-fold cross-validation:

### Average Accuracy

The average accuracy represents the overall performance across all folds.

### Highest Accuracy

The highest accuracy shows the best performance among the folds.

### Lowest Accuracy

The lowest accuracy shows the weakest performance among the folds.

This helps me understand how consistently the model performs.

---

## Comparing Different Numbers of Folds

I also tested different values of `cv`:

- 3-fold cross-validation
- 5-fold cross-validation
- 10-fold cross-validation

The purpose is to understand how changing the number of folds affects model evaluation.

For very small datasets, a large number of folds may not be appropriate because there may not be enough samples in each class.

---

## Optional Challenge

For the optional challenge, I compared:

- Decision Tree
- Random Forest

Both models were evaluated using **5-fold cross-validation**.

The average cross-validation accuracy of each model was calculated.

The program then determines which model performed better.

---

## Model Comparison

The comparison follows this idea:

- If Random Forest has higher average accuracy, Random Forest performed better.
- If Decision Tree has higher average accuracy, Decision Tree performed better.
- If both have the same accuracy, both models performed equally.

The goal is not simply to make one model win. The goal is to understand how different models perform on the same dataset.

---

## Technologies Used

- Python
- Pandas
- Scikit-learn

### Scikit-learn Components

- `train_test_split`
- `RandomForestClassifier`
- `DecisionTreeClassifier`
- `accuracy_score`
- `cross_val_score`

---

## Project Structure

    random-forest-cross-validation/
    ├── student_data.csv
    ├── day19_cross_validation.py
    └── README.md

---

## What I Learned

From this project, I learned:

- A single train/test split may not always give a reliable evaluation.
- Cross-validation evaluates a model using multiple data splits.
- K-Fold Cross-Validation divides data into several folds.
- `cross_val_score()` makes cross-validation easier in Scikit-learn.
- The average cross-validation score can give a better idea of model performance.
- Different numbers of folds can produce different results.
- Decision Trees and Random Forests can be compared using the same evaluation method.
- Random Forest uses multiple Decision Trees to make predictions.

---

## Key Takeaway

The main lesson from Day 19 is:

**Cross-validation provides a more reliable estimate of model performance by evaluating the model on multiple train/test splits.**

Instead of judging a model only from one train/test split, cross-validation helps us see how consistently the model performs.

---

## Future Improvements

In future projects, I can improve this work by:

- Testing additional machine learning models
- Using more evaluation metrics such as precision, recall, and F1-score
- Performing hyperparameter tuning
- Using a larger dataset
- Creating visualizations of cross-validation results
- Comparing more machine learning algorithms

---

## Author

Abone Alam

Day 19 of my Machine Learning learning journey.