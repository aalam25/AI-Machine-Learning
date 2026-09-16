import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)



# Load Dataset

df = pd.read_csv("student_data.csv")

print("First 5 Students:")
print(df.head())



# Create Features and Target

X = df[
    [
        "Study_Hours",
        "Attendance",
        "Assignment_Score",
        "Midterm_Score",
        "Exam_Score"
    ]
]

y = df["Final_Result"].map({
    "Pass": 1,
    "Fail": 0
})



# Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# Train Random Forest Model

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)



# Make Predictions

predictions = model.predict(X_test)



# Model Evaluation

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions
)

recall = recall_score(
    y_test,
    predictions
)

f1 = f1_score(
    y_test,
    predictions
)


print("\nModel Evaluation Summary")
print("------------------------")

print(f"Accuracy:  {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1 Score:  {f1:.2f}")



# Classification Report

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions
    )
)



# Confusion Matrix

cm = confusion_matrix(
    y_test,
    predictions
)

print("\nConfusion Matrix:")
print(cm)



# Class Distribution

print("\nClass Distribution:")
print(y.value_counts())

print("\nClass Distribution Percentage:")
print(y.value_counts(normalize=True))



# Actual vs Predicted

print("\nActual vs Predicted:")

for actual, predicted in zip(
    y_test,
    predictions
):
    print(
        f"Actual: {actual} | Predicted: {predicted}"
    )



# New Student Prediction

new_student = np.array([[
    6,
    90,
    82,
    85,
    88
]])

prediction = model.predict(
    new_student
)

probability = model.predict_proba(
    new_student
)

print("\nNew Student Prediction:")

if prediction[0] == 1:
    print("Pass")
else:
    print("Fail")

print(
    f"Pass Probability: {probability[0][1]:.2f}"
)