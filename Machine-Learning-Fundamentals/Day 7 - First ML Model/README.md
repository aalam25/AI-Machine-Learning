# First Machine Learning Model: Exam Score Prediction

## Description

This project is my first machine learning project using Python and Scikit-learn.

The goal of the project is to train a Linear Regression model to predict a student's exam score based on the number of hours they studied.

The project demonstrates the basic machine learning workflow, including preparing data, splitting data into training and testing sets, training a model, making predictions, and evaluating the model.

## Dataset

The project uses a small sample dataset containing:

- Study Hours
- Exam Scores

### Example

| Study Hours | Exam Score |
|------------:|-----------:|
| 1 | 50 |
| 2 | 55 |
| 3 | 60 |
| 4 | 65 |
| 5 | 70 |
| 6 | 78 |
| 7 | 85 |
| 8 | 92 |

## Machine Learning Approach

This project uses **Linear Regression**.

The model learns the relationship between:

- **Feature:** Study Hours
- **Target:** Exam Score

### Machine Learning Workflow

```text
Dataset
   ↓
Features and Target
   ↓
Train/Test Split
   ↓
Linear Regression Model
   ↓
Model Training
   ↓
Predictions
   ↓
Model Evaluation

```



## Features

The program:

1. Creates a dataset using NumPy
2. Separates features and target variables
3. Splits the dataset into training and testing data
4. Creates a Linear Regression model
5. Trains the model using training data
6. Makes predictions on test data
7. Calculates Mean Absolute Error (MAE)
8. Calculates R² Score
9. Displays the model coefficient and intercept
10. Predicts an exam score for a new student




## Model Evaluation:

The project uses two evaluation metrics:

Mean Absolute Error (MAE):
MAE measures the average difference between the actual and predicted values.
A lower MAE generally indicates better predictions.


R² Score:
R² measures how well the model explains the variation in the target values.
A value closer to 1 generally indicates a stronger model fit.




## Example Prediction:
The trained model is used to predict the exam score of a student who studies for 10 hours.
Study Hours: 10
Predicted Exam Score: [model prediction]




## Technologies Used:
Python
NumPy
Scikit-learn




## What I Learned:
1. Through this project, I learned the basic workflow of supervised machine learning.
2. I learned the difference between features and targets, how to divide data into training and testing sets, and how a Linear Regression model can learn a relationship between variables.
3. I also learned how to evaluate a machine learning model using Mean Absolute Error and R² Score and how to use a trained model to make predictions on new data.




## Project Purpose:
This project is part of my learning journey toward Artificial Intelligence and Machine Learning.
It is my first machine learning project and builds on the Python, NumPy, Pandas, data cleaning, visualization, and statistics skills I developed in previous projects.