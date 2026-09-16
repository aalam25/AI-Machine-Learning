import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


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


# Create Random Forest Model

model = RandomForestClassifier(
    random_state=42
)


# Create Parameter Grid

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 3, 5, 10],
    "min_samples_split": [2, 5]
}


# Create GridSearchCV

grid_search = GridSearchCV(
    model,
    param_grid,
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

print("\nBest Cross-Validation Accuracy:")
print(grid_search.best_score_)


# Accuracy

best_model = grid_search.best_estimator_

predictions = best_model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nTuned Model Accuracy:")
print(f"{accuracy:.2f}")


# Precision

precision = precision_score(
    y_test,
    predictions
)

print("\nPrecision:")
print(f"{precision:.2f}")


# Recall

recall = recall_score(
    y_test,
    predictions
)

print("\nRecall:")
print(f"{recall:.2f}")


# F1-Score

f1 = f1_score(
    y_test,
    predictions
)

print("\nF1-Score:")
print(f"{f1:.2f}")


# Display All Metrics

print("\nModel Evaluation:")
print("-----------------")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")


# Confusion Matrix

cm = confusion_matrix(
    y_test,
    predictions
)

print("\nConfusion Matrix:")
print(cm)


# Understand the Confusion Matrix

print("\nConfusion Matrix Explanation:")
print("TN = Student predicted Fail and actually Fail")
print("FP = Student predicted Pass but actually Fail")
print("FN = Student predicted Fail but actually Pass")
print("TP = Student predicted Pass and actually Pass")


# Classification Report

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions
))


# Train Basic Random Forest

basic_model = RandomForestClassifier(
    n_estimators=100,
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


# Compare Basic and Tuned Models

print("\nFinal Model Comparison:")
print("-----------------------")

print(f"Basic Random Forest Accuracy: {basic_accuracy:.2f}")
print(f"Tuned Random Forest Accuracy: {accuracy:.2f}")

if accuracy > basic_accuracy:
    print("Tuned model performed better.")
elif basic_accuracy > accuracy:
    print("Basic model performed better.")
else:
    print("Both models performed equally.")