# Random Forest Hyperparameter Tuning with GridSearchCV

## Overview

This project focuses on improving a Random Forest classification model by using hyperparameter tuning with GridSearchCV.

The goal is to predict whether a student will Pass or Fail based on study hours, attendance, assignment score, and exam score.

Instead of using only one set of Random Forest settings, GridSearchCV tests multiple combinations of hyperparameters and uses cross-validation to identify the combination that performs best on the training data.

The project also compares the basic Random Forest model with the tuned Random Forest model on the test dataset.

## Objective

The main objectives of this project are to:

- Understand hyperparameters in machine learning
- Train a basic Random Forest classifier
- Use GridSearchCV for hyperparameter tuning
- Test different Random Forest hyperparameter combinations
- Use 5-fold cross-validation
- Identify the best hyperparameters
- Evaluate the tuned model on unseen test data
- Compare the basic and tuned Random Forest models
- Understand why hyperparameter tuning does not always improve test accuracy

## Dataset

The project uses a student dataset named `student_data.csv`.

The dataset contains information about students' academic performance and final results.

### Features Used

The model uses the following four features:

- `Study_Hours` - Number of hours the student studies
- `Attendance` - Student attendance percentage
- `Assignment_Score` - Student assignment score
- `Exam_Score` - Student exam score

### Target Variable

The target variable is `Final_Result`.

The target is converted into numerical values:

- `Pass` = 1
- `Fail` = 0

## Machine Learning Model

This project uses the Random Forest Classifier.

Random Forest is an ensemble machine learning algorithm that combines multiple decision trees to make predictions.

Each decision tree learns from the training data, and the Random Forest combines the results of multiple trees to make a final prediction.

## Train-Test Split

The dataset is divided into training and testing data using an 80/20 split.

- 80% of the data is used for training
- 20% of the data is used for testing

The split uses `random_state=42` so that the results can be reproduced.

In this project:

- Training data: 40 students
- Testing data: 10 students

## Basic Random Forest Model

Before tuning the model, a basic Random Forest classifier is trained.

The basic model uses:

- `random_state=42`
- Default Random Forest hyperparameters

The basic Random Forest achieved:

**Test Accuracy: 90%**

## What Are Hyperparameters?

Hyperparameters are settings that are chosen before a machine learning model is trained.

For Random Forest, some important hyperparameters include:

### n_estimators

This controls the number of decision trees in the Random Forest.

For example:

- 50 trees
- 100 trees
- 200 trees

Generally, more trees can make a model more stable, but they can also increase training time.

### max_depth

This controls the maximum depth of each decision tree.

Possible values in this project include:

- `None`
- 3
- 5
- 10

A value of `None` allows the tree to continue growing until other stopping conditions are reached.

### min_samples_split

This controls the minimum number of samples required to split an internal node.

Values tested in this project include:

- 2
- 5

## Parameter Grid

The project uses the following parameter grid:

- `n_estimators`: 50, 100, 200
- `max_depth`: None, 3, 5, 10
- `min_samples_split`: 2, 5

This creates multiple possible combinations of Random Forest settings that GridSearchCV can evaluate.

## GridSearchCV

GridSearchCV is used to automatically test different combinations of hyperparameters.

Instead of manually trying different settings, GridSearchCV evaluates the combinations and identifies the best-performing configuration.

The model uses:

- 5-fold cross-validation
- Accuracy as the scoring metric

## Cross-Validation

The project uses 5-fold cross-validation.

The training data is divided into five parts.

The model is trained and evaluated multiple times using different parts of the training data.

This helps provide a more reliable estimate of how well different hyperparameter combinations perform.

## Best Hyperparameters

GridSearchCV selected the following hyperparameters as the best combination:

- `n_estimators = 50`
- `max_depth = None`
- `min_samples_split = 2`

The best model was:

`RandomForestClassifier(n_estimators=50, random_state=42)`

## Best Cross-Validation Accuracy

The best cross-validation accuracy was:

**95%**

This means the best hyperparameter combination achieved an average accuracy of 95% during the 5-fold cross-validation process.

## Tuned Model Test Accuracy

After selecting the best hyperparameters, the tuned model was evaluated on the separate test dataset.

The tuned Random Forest achieved:

**90% test accuracy**

The test dataset was not used to select the best hyperparameters, which makes it useful for evaluating the model's performance on unseen data.

## Model Comparison

The project compares the basic Random Forest with the tuned Random Forest.

### Basic Random Forest

Test Accuracy:

**90%**

### Tuned Random Forest

Test Accuracy:

**90%**

### Accuracy Improvement

**0%**

Both models performed equally on the test dataset.

## Results

The final results of the project are:

- Basic Random Forest Accuracy: **90%**
- Best Cross-Validation Accuracy: **95%**
- Tuned Random Forest Accuracy: **90%**
- Accuracy Improvement: **0%**

The tuned model did not improve the test accuracy compared with the basic Random Forest.

## Why Did Tuning Not Improve Test Accuracy?

Hyperparameter tuning does not always guarantee better test performance.

In this project, the basic Random Forest was already performing well, achieving 90% test accuracy.

The dataset contains only 50 students, which is relatively small for machine learning. Because the test set contains only 10 students, even one incorrect prediction changes the test accuracy by 10 percentage points.

Therefore, the tuned model and basic model can have the same test accuracy even though GridSearchCV found a different set of hyperparameters.

The 95% cross-validation score and 90% test accuracy also show that performance on the training data through cross-validation does not necessarily equal performance on completely unseen test data.

## Important Lesson

One of the most important lessons from this project is that **hyperparameter tuning is not guaranteed to improve test accuracy**.

The purpose of tuning is to find a good configuration based on the available training data and cross-validation.

The final model should always be evaluated on a separate test dataset.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Random Forest
- GridSearchCV
- Cross-Validation
- Accuracy Score

## Python Libraries

The project uses:

- `pandas`
- `sklearn.model_selection`
- `sklearn.ensemble`
- `sklearn.metrics`

## Project Structure

The project contains the following files:

- `student_data.csv` - Student dataset
- `day20_grid_search.py` - Python machine learning program
- `README.md` - Project documentation

## How to Run the Project

### Step 1: Install Required Libraries

Make sure Python is installed on your computer.

Install the required libraries using:

`pip install pandas scikit-learn`

### Step 2: Place the Dataset

Make sure `student_data.csv` is in the same folder as `day20_grid_search.py`.

### Step 3: Run the Program

Run:

`python day20_grid_search.py`

The program will display:

- Dataset
- Selected features
- Target values
- Training data shape
- Testing data shape
- Basic Random Forest accuracy
- Parameter grid
- Best hyperparameters
- Best cross-validation accuracy
- Best Random Forest model
- Tuned Random Forest accuracy
- Model comparison
- Accuracy improvement

## Example Results

The results from this project were:

Basic Random Forest Accuracy: 0.90

Best Parameters:

`{'max_depth': None, 'min_samples_split': 2, 'n_estimators': 50}`

Best Cross-Validation Accuracy:

`0.95`

Tuned Random Forest Accuracy:

`0.90`

Final Model Comparison:

Basic Random Forest Accuracy: 0.90

Tuned Random Forest Accuracy: 0.90

Result:

Both models performed equally.

Accuracy Improvement:

`0.00`

## What I Learned

Through this project, I learned:

- What hyperparameters are
- Why hyperparameters are important
- How Random Forest hyperparameters affect a model
- How to create a parameter grid
- How GridSearchCV works
- How 5-fold cross-validation works
- How to find the best hyperparameters
- How to use `best_params_`
- How to use `best_score_`
- How to use `best_estimator_`
- How to evaluate a tuned model on test data
- How to compare basic and tuned models
- Why cross-validation accuracy and test accuracy can be different
- Why hyperparameter tuning does not always improve test accuracy

## Key Takeaway

The main lesson from Day 20 is that **GridSearchCV can automatically search for better hyperparameter combinations, but the best cross-validation result does not guarantee higher test accuracy**.

In this project, GridSearchCV found a model with 95% cross-validation accuracy, but the tuned model achieved the same 90% test accuracy as the basic Random Forest.

This shows why it is important to separate training data from test data and evaluate the final model using unseen data.

## Future Improvements

Possible improvements for this project include:

- Using a larger dataset
- Testing additional Random Forest hyperparameters
- Comparing Random Forest with other machine learning algorithms
- Using additional evaluation metrics
- Performing more extensive cross-validation
- Visualizing model performance
- Evaluating precision, recall, and F1-score
- Creating a confusion matrix
- Saving the best trained model
- Building a student result prediction application

## Project Context

This project is part of my ongoing Artificial Intelligence and Machine Learning learning journey.

It builds on previous projects involving:

- Python
- NumPy
- Pandas
- Data Analysis
- Data Visualization
- Decision Trees
- Random Forest
- Cross-Validation
- Model Evaluation

The project helps develop practical experience with machine learning model improvement and evaluation.

## Author

**Abone Alam**

Computer Science Student | Aspiring AI/ML Professional