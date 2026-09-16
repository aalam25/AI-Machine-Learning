import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



# Load Dataset

df = pd.read_csv("student_data.csv")



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



# 3. Create Target (y)

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

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)



# Basic Random Forest Model

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

print("\nBasic Random Forest Accuracy:")
print(f"{basic_accuracy:.2f}")



# Test Different Max Depth Values

print("\nTesting Different Max Depth Values:")

for depth in [2, 3, 5, 10, None]:

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=depth,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        f"Max Depth: {depth} | "
        f"Accuracy: {accuracy:.2f}"
    )



# Test Different Numbers of Trees

print("\nRandom Forest n_estimators Comparison:")

for trees in [10, 50, 100, 200, 300]:

    forest = RandomForestClassifier(
        n_estimators=trees,
        random_state=42
    )

    forest.fit(
        X_train,
        y_train
    )

    tree_predictions = forest.predict(
        X_test
    )

    tree_accuracy = accuracy_score(
        y_test,
        tree_predictions
    )

    print(
        f"Trees: {trees} | "
        f"Accuracy: {tree_accuracy:.2f}"
    )



# Find Best Max Depth

best_accuracy = 0
best_depth = None

for depth in [2, 3, 5, 10, None]:

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=depth,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_depth = depth


print("\nBest Model Configuration:")
print(f"Best Max Depth: {best_depth}")
print(f"Best Accuracy: {best_accuracy:.2f}")



# Train Improved Model

best_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=best_depth,
    random_state=42
)

best_model.fit(
    X_train,
    y_train
)

improved_predictions = best_model.predict(
    X_test
)

improved_accuracy = accuracy_score(
    y_test,
    improved_predictions
)

print("\nImproved Model Accuracy:")
print(f"{improved_accuracy:.2f}")



# Compare Basic and Improved Models

print("\nModel Comparison:")
print("-------------------------")

print(
    f"Basic Model Accuracy:    "
    f"{basic_accuracy:.2f}"
)

print(
    f"Improved Model Accuracy: "
    f"{improved_accuracy:.2f}"
)


if improved_accuracy > basic_accuracy:

    print("\nImproved model performed better.")

elif basic_accuracy > improved_accuracy:

    print("\nBasic model performed better.")

else:

    print("\nBoth models achieved the same accuracy.")