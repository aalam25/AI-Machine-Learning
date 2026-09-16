import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


# Load Dataset

df = pd.read_csv("student_data.csv")

print("\nDataset:")
print(df)


# Select Features

X = df[
    [
        "Study_Hours",
        "Attendance",
        "Assignment_Score",
        "Exam_Score"
    ]
]

print("\nFeatures:")
print(X)


# Create Target

y = df["Final_Result"].map({
    "Pass": 1,
    "Fail": 0
})

print("\nTarget:")
print(y)


# Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)


# Create Basic Random Forest Model

basic_model = RandomForestClassifier(
    random_state=42
)

basic_model.fit(
    X_train,
    y_train
)

basic_predictions = basic_model.predict(
    X_test
)

basic_accuracy = accuracy_score(
    y_test,
    basic_predictions
)

print("\nBasic Random Forest Accuracy:")
print(f"{basic_accuracy:.2f}")


# Create Parameter Grid

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 3, 5, 10],
    "min_samples_split": [2, 5]
}

print("\nParameter Grid:")
print(param_grid)


# Create GridSearchCV

grid_search = GridSearchCV(
    estimator=basic_model,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy"
)


# Train GridSearchCV

grid_search.fit(
    X_train,
    y_train
)


# Display Best Parameters

print("\nBest Parameters:")
print(grid_search.best_params_)


# Display Best Cross-Validation Score

print("\nBest Cross-Validation Accuracy:")
print(f"{grid_search.best_score_:.2f}")


# Get Best Model

best_model = grid_search.best_estimator_

print("\nBest Model:")
print(best_model)


# Make Predictions Using Tuned Model

tuned_predictions = best_model.predict(
    X_test
)


# Calculate Tuned Model Accuracy

tuned_accuracy = accuracy_score(
    y_test,
    tuned_predictions
)

print("\nTuned Random Forest Accuracy:")
print(f"{tuned_accuracy:.2f}")


# Compare Basic and Tuned Models

print("\nModel Comparison:")
print("-----------------")

print(f"Basic Random Forest Accuracy: {basic_accuracy:.2f}")
print(f"Tuned Random Forest Accuracy: {tuned_accuracy:.2f}")


# Determine Which Model Performed Better

if tuned_accuracy > basic_accuracy:
    print("\nResult:")
    print("The tuned Random Forest performed better.")

elif basic_accuracy > tuned_accuracy:
    print("\nResult:")
    print("The basic Random Forest performed better.")

else:
    print("\nResult:")
    print("Both models performed equally.")


# Check Improvement

improvement = tuned_accuracy - basic_accuracy

print("\nAccuracy Improvement:")
print(f"{improvement:.2f}")