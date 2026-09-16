# Random Forest Hyperparameter Tuning and Model Comparison

## Description

This project is part of my machine learning learning journey.

The goal of this project is to improve a Random Forest classification model by testing different hyperparameters and comparing the performance of a basic model with an improved model.

The model predicts whether a student will pass or fail based on their academic information.

The features used in the model are:

- Study Hours
- Attendance
- Assignment Score
- Midterm Score
- Exam Score

The target variable is the student's final result.

## Dataset

The project uses a student dataset stored in:

student_data.csv

The dataset contains information about students and their academic performance.

### Features

- Study_Hours - Number of hours the student studies
- Attendance - Student attendance percentage
- Assignment_Score - Assignment score
- Midterm_Score - Midterm exam score
- Exam_Score - Exam score

### Target Variable

The target variable is:

Final_Result

The target is converted into numerical values:

- Pass = 1
- Fail = 0

## Machine Learning Approach

This project uses the Random Forest Classifier.

Random Forest is an ensemble machine learning algorithm that combines multiple decision trees to make predictions.

The workflow of this project is:

Student Dataset
       ↓
Select Features
       ↓
Create Target
       ↓
Train/Test Split
       ↓
Train Random Forest
       ↓
Test Hyperparameters
       ↓
Find Best Configuration
       ↓
Train Improved Model
       ↓
Compare Models

## Train/Test Split

The dataset is divided into training and testing data using train_test_split().

The project uses:

test_size = 0.2

This means:

- 80% of the data is used for training
- 20% of the data is used for testing

The project also uses:

random_state = 42

This makes the data split reproducible.

## Basic Random Forest Model

The basic Random Forest model uses 100 trees.

The model is trained using the training data and evaluated using the testing data.

## Hyperparameter Tuning

Hyperparameters are settings that control how a machine learning model works.

This project tests two important Random Forest hyperparameters:

- max_depth
- n_estimators

## Testing max_depth

The max_depth parameter controls the maximum depth of each decision tree.

The project tests the following values:

- 2
- 3
- 5
- 10
- None

A smaller max_depth creates simpler trees.

A larger max_depth allows the trees to become more complex.

The project compares the accuracy of each value.

## Testing n_estimators

The n_estimators parameter controls the number of decision trees used by the Random Forest.

The project tests:

- 10
- 50
- 100
- 200
- 300

Using more trees can make the model more stable, but it can also increase training time.

## Finding the Best max_depth

The project automatically searches for the best max_depth based on test accuracy.

The program starts with a best accuracy of 0 and tests each max_depth value.

If a better accuracy is found, the program saves that configuration.

The program then displays the best configuration.

Example:

Best Model Configuration:
Best Max Depth: 5
Best Accuracy: 0.85

The actual values depend on the dataset.

## Improved Random Forest Model

After finding the best max_depth, the project creates an improved Random Forest model.

The improved model uses:

- n_estimators = 100
- max_depth = best_depth
- random_state = 42

The improved model is then trained and evaluated on the test data.

## Model Comparison

The project compares two Random Forest models.

### Basic Model

The basic model uses:

- n_estimators = 100
- max_depth = default

### Improved Model

The improved model uses:

- n_estimators = 100
- max_depth = best_depth

The accuracy of both models is compared.

The program determines whether:

- The improved model performed better
- The basic model performed better
- Both models achieved the same accuracy

## Accuracy

Accuracy measures how many predictions the model classified correctly.

Accuracy = Correct Predictions / Total Predictions

For example, an accuracy of 0.80 means the model correctly classified 80% of the test examples.

A higher accuracy generally indicates better performance on the test dataset.

## Example Output

The actual output depends on the dataset.

Example:

Testing Different Max Depth Values:

Max Depth: 2 | Accuracy: 0.75
Max Depth: 3 | Accuracy: 0.80
Max Depth: 5 | Accuracy: 0.85
Max Depth: 10 | Accuracy: 0.80
Max Depth: None | Accuracy: 0.80

Random Forest Comparison:

Trees: 10 | Accuracy: 0.75
Trees: 50 | Accuracy: 0.80
Trees: 100 | Accuracy: 0.80
Trees: 200 | Accuracy: 0.80
Trees: 300 | Accuracy: 0.80

Best Model Configuration:

Best Max Depth: 5
Best Accuracy: 0.85

Improved Model Accuracy:
0.85

Model Comparison:
----------------

Basic Model Accuracy: 0.80
Improved Model Accuracy: 0.85

Improved model performed better.

These numbers are examples only. The actual results are generated when the program runs.

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Random Forest
- Machine Learning

## Libraries Used

The project uses the following Python libraries:

- pandas
- numpy
- scikit-learn

## What I Learned

Through this project, I learned how hyperparameters can affect the performance of a machine learning model.

I learned:

- What hyperparameters are
- How max_depth affects decision trees
- How n_estimators controls the number of trees
- How to test different hyperparameter values
- How to find the best max_depth
- How to train an improved Random Forest model
- How to compare a basic model with an improved model
- How to use accuracy to compare model performance
- Why model complexity can affect overfitting
- Why testing different configurations is important in machine learning

## Overfitting

Overfitting happens when a model learns the training data too closely and does not perform as well on new data.

Very deep decision trees can become too complex and may increase the risk of overfitting.

The max_depth parameter can help control the complexity of the trees.

Testing different values helps find a configuration that performs better on unseen test data.

## Project Structure

random-forest-model-improvement/
│
├── student_data.csv
├── random_forest_improvement.py
└── README.md

## How to Run

### Step 1: Install Python

Make sure Python is installed on your computer.

### Step 2: Install Required Libraries

Open a terminal and run:

pip install pandas numpy scikit-learn

### Step 3: Add the Dataset

Make sure student_data.csv is in the same folder as the Python file.

### Step 4: Run the Program

Run:

python random_forest_improvement.py

The program will display:

- Training data shape
- Testing data shape
- Accuracy for different max_depth values
- Accuracy for different numbers of trees
- Best model configuration
- Improved model accuracy
- Basic model accuracy
- Model comparison

## Future Improvements

In the future, this project could be improved by:

- Using GridSearchCV for automated hyperparameter tuning
- Using cross-validation
- Testing more Random Forest hyperparameters
- Comparing Random Forest with Logistic Regression and other models
- Using a larger student dataset
- Adding precision, recall, and F1-score
- Adding confusion matrix visualization
- Adding feature importance visualization
- Building a simple user interface for student predictions

## Project Purpose

This project is part of my learning journey toward Artificial Intelligence and Machine Learning.

It builds on the skills I developed in previous projects, including:

- Python
- NumPy
- Pandas
- Data Analysis
- Statistics
- Data Visualization
- Linear Regression
- Logistic Regression
- Decision Trees
- Random Forest

This project helped me understand how machine learning models can be improved through hyperparameter tuning and model comparison.

## Conclusion

This project demonstrates how a Random Forest classification model can be improved by testing different hyperparameter values.

By experimenting with max_depth and n_estimators, I learned how model settings can affect prediction performance.

I also learned how to find a better model configuration, train an improved model, and compare it with a basic Random Forest model.

This project is another step in my machine learning learning journey and gives me practical experience with model optimization, hyperparameter tuning, and model evaluation.