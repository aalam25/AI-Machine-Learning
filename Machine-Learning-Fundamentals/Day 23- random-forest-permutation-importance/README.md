# Random Forest Permutation Feature Importance

## Project Overview

This project demonstrates how to use **Permutation Feature Importance** with a Random Forest classification model.

The goal is to understand which student features have the greatest effect on the model's predictions and to compare **Permutation Feature Importance** with the traditional Random Forest `feature_importances_` method.

This is **Day 23** of my Machine Learning learning journey.

## Project Goal

The project uses student academic data to predict whether a student will **Pass** or **Fail**.

The main objectives are:

- Train a Random Forest classification model.
- Calculate traditional Random Forest feature importance.
- Calculate permutation feature importance.
- Compare the two feature-importance methods.
- Visualize permutation importance.
- Identify the feature with the highest permutation importance.

## Project Files

    random-forest-permutation-importance/
    │
    ├── permutation_importance.py
    ├── student_data.csv
    ├── permutation_importance.png
    └── README.md

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Random Forest
- Permutation Feature Importance
- Git & GitHub

## Dataset

The dataset contains **50 student records** and the following columns:

| Column | Description |
|---|---|
| `Student_ID` | Unique student identifier |
| `Study_Hours` | Number of hours studied |
| `Attendance` | Student attendance percentage |
| `Assignment_Score` | Assignment score |
| `Midterm_Score` | Midterm examination score |
| `Exam_Score` | Final examination score |
| `Final_Result` | Pass or Fail |

For the machine learning model, the following four features were used:

- Study_Hours
- Attendance
- Assignment_Score
- Exam_Score

The target variable was:

- `Pass` → 1
- `Fail` → 0

## Machine Learning Process

The project follows these steps:

1. Load the dataset.
2. Select the input features.
3. Convert the target labels into numerical values.
4. Split the dataset into training and testing data.
5. Train a Random Forest classifier.
6. Calculate traditional feature importance.
7. Calculate permutation feature importance.
8. Create a visualization.
9. Identify the feature with the highest permutation importance.

## Random Forest Model

The model was created using:

    RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

The dataset was divided into:

- Training data: `(40, 4)`
- Testing data: `(10, 4)`

The Random Forest model was successfully trained.

## Traditional Feature Importance

Random Forest provides feature importance through:

    model.feature_importances_

The results were:

| Feature | Importance |
|---|---:|
| Exam_Score | 0.355541 |
| Attendance | 0.280771 |
| Assignment_Score | 0.278750 |
| Study_Hours | 0.084938 |

According to traditional Random Forest feature importance, **Exam_Score** had the highest importance.

## Permutation Feature Importance

Permutation Feature Importance works differently.

It randomly shuffles one feature at a time and observes how the model's performance changes.

The calculation used:

    permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="accuracy"
    )

The results were:

| Feature | Permutation Importance |
|---|---:|
| Study_Hours | 0.00 |
| Assignment_Score | 0.00 |
| Attendance | -0.05 |
| Exam_Score | -0.05 |

The feature with the highest measured permutation importance was:

**Study_Hours — 0.0000**

However, this does **not** mean Study_Hours was strongly important. It means that shuffling Study_Hours did not change the model's measured accuracy on this particular test set.

## Visualization

The project creates a bar chart showing permutation feature importance.

The chart is saved as:

    permutation_importance.png

The visualization makes it easier to compare the effect of shuffling each feature.

## Traditional vs. Permutation Importance

The two methods produced different results.

### Traditional Feature Importance

The Random Forest ranked:

1. Exam_Score
2. Attendance
3. Assignment_Score
4. Study_Hours

### Permutation Feature Importance

The results were:

1. Study_Hours — 0.00
2. Assignment_Score — 0.00
3. Attendance — -0.05
4. Exam_Score — -0.05

This difference is expected because the two methods measure feature importance in different ways.

Traditional Random Forest importance is based on how features contribute to splitting the decision trees.

Permutation importance measures how much the model's performance changes when a feature is randomly shuffled.

## Why Are Some Permutation Importance Values Negative?

Permutation importance can sometimes produce negative values.

A negative value means that the model performed slightly **better** after the feature was shuffled.

This does not necessarily mean that the feature is harmful.

In this project, the test set contains only **10 samples**, so small changes in predictions can have a noticeable effect on accuracy.

Therefore, the negative values of:

    Attendance       -0.05
    Exam_Score       -0.05

should not be interpreted as proof that these features are harmful.

## Important Lesson

This project taught me that feature importance methods can give different results.

A feature can have relatively high traditional Random Forest importance but show little or even negative permutation importance on a small test set.

Permutation importance should therefore be interpreted together with:

- Model performance
- Dataset size
- Test set size
- Feature relationships
- Other feature-importance methods

It is also important to remember that **feature importance does not prove causation**.

## Key Concepts Learned

### 1. Traditional Feature Importance

    model.feature_importances_

This measures how much each feature contributes to the Random Forest's tree-based decisions.

### 2. Permutation Importance

    permutation_importance()

This measures how model performance changes when a feature is randomly shuffled.

### 3. `importances_mean`

    permutation.importances_mean

This gives the average importance across the repeated shuffling experiments.

### 4. `n_repeats`

    n_repeats=10

The feature is shuffled multiple times to obtain a more stable estimate.

### 5. Negative Importance

Permutation importance can be negative, especially with small datasets or test sets.

## How to Run the Project

### Step 1: Clone the repository

    git clone https://github.com/aalam25/random-forest-permutation-importance.git

### Step 2: Open the project folder

    cd random-forest-permutation-importance

### Step 3: Install the required libraries

    pip install pandas scikit-learn matplotlib

### Step 4: Run the Python program

    python permutation_importance.py

The program will:

- Load the dataset
- Train the Random Forest model
- Calculate traditional feature importance
- Calculate permutation importance
- Display the results
- Create the permutation importance graph

## Example Output

    Training Data Shape:
    (40, 4)

    Testing Data Shape:
    (10, 4)

    Random Forest Model Trained Successfully.

    Traditional Feature Importance:
                Feature  Importance
    3        Exam_Score    0.355541
    1        Attendance    0.280771
    2  Assignment_Score    0.278750
    0       Study_Hours    0.084938

    Permutation Feature Importance:
                Feature  Importance
    0       Study_Hours        0.00
    2  Assignment_Score        0.00
    1        Attendance       -0.05
    3        Exam_Score       -0.05

    Most Important Feature According to Permutation Importance:
    Study_Hours

    Importance Value:
    0.0000

## What I Learned

Through this project, I learned how to:

- Use Random Forest for classification.
- Calculate traditional feature importance.
- Calculate permutation feature importance.
- Understand how feature shuffling affects model performance.
- Compare different feature-importance methods.
- Create feature-importance visualizations.
- Interpret zero and negative permutation importance values.
- Understand the limitations of feature importance on small datasets.

## Machine Learning Journey

This project is part of my ongoing Machine Learning portfolio.

Previous projects covered:

- Decision Tree Classification
- Random Forest Classification
- Model Evaluation
- Confusion Matrix
- Feature Importance
- Decision Tree vs. Random Forest
- Hyperparameter Tuning
- Cross-Validation
- GridSearchCV
- Random Forest Evaluation
- Random Forest Feature Importance

This project builds on those concepts by exploring **Permutation Feature Importance**.

## Future Improvements

Possible improvements for this project include:

- Use a larger dataset.
- Increase the size of the test set.
- Perform more permutation repeats.
- Compare permutation importance across cross-validation folds.
- Compare additional machine learning models.
- Explore SHAP values for more advanced model interpretation.

## Author

**Abone Alam**

Computer Science Student | Aspiring Machine Learning / AI Professional

GitHub: https://github.com/aalam25

## Conclusion

This project helped me understand that looking at feature importance from only one method may not give the complete picture.

Traditional Random Forest importance and permutation importance answer different questions. Comparing both methods provides a better understanding of how a machine learning model uses its features.

This project strengthened my understanding of **model interpretation and explainable machine learning**.