import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix



# Load Dataset

df = pd.read_csv("student-data.csv")

print("First 5 Students:")
print(df.head())



# Create Features (X)

X = df[
    [
        "Study_Hours",
        "Attendance",
        "Assignment_Score",
        "Midterm_Score",
        "Exam_Score"
    ]
]

print("\nFeatures:")
print(X)



# Create Target (y)

y = df["Final_Result"].map({
    "Pass": 1,
    "Fail": 0
})

print("\nTarget:")
print(y)



# Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\nTraining Features Shape:")
print(X_train.shape)

print("\nTesting Features Shape:")
print(X_test.shape)

print("\nTraining Target Shape:")
print(y_train.shape)

print("\nTesting Target Shape:")
print(y_test.shape)



# Create Decision Tree Model

model = DecisionTreeClassifier(
    random_state=42
)



# Train Model

model.fit(X_train, y_train)



# Make Predictions

predictions = model.predict(X_test)

print("\nPredictions:")
print(predictions)



# Model Accuracy

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nAccuracy:")
print(accuracy)



# Actual vs Predicted

print("\nActual vs Predicted:")

for actual, predicted in zip(y_test, predictions):

    print(
        f"Actual: {actual} | Predicted: {predicted}"
    )



# Confusion Matrix

cm = confusion_matrix(
    y_test,
    predictions
)

print("\nConfusion Matrix:")
print(cm)



# Predict One New Student

new_student = np.array([[
    6,      # Study Hours
    90,     # Attendance
    82,     # Assignment Score
    85,     # Midterm Score
    88      # Exam Score
]])


prediction = model.predict(new_student)


print("\nNew Student Prediction:")

if prediction[0] == 1:
    print("Pass")
else:
    print("Fail")



# Predict Multiple New Students

students = np.array([
    [2, 65, 55, 50, 52],
    [5, 82, 75, 70, 72],
    [8, 95, 92, 90, 94]
])


student_predictions = model.predict(students)


print("\nNew Student Predictions:")

for i, prediction in enumerate(student_predictions, start=1):

    if prediction == 1:
        result = "Pass"
    else:
        result = "Fail"

    print(f"Student {i}: {result}")



# Decision Tree Depth Comparison

print("\nDecision Tree Depth Comparison:")

for depth in [2, 3, 5]:

    tree_model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    tree_model.fit(
        X_train,
        y_train
    )

    depth_predictions = tree_model.predict(
        X_test
    )

    depth_accuracy = accuracy_score(
        y_test,
        depth_predictions
    )

    print(
        f"Max Depth: {depth} | "
        f"Accuracy: {depth_accuracy:.2f}"
    )