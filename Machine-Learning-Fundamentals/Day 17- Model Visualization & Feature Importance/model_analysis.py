import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix



# Load Dataset

df = pd.read_csv("student_data.csv")

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

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)



# Random Forest Model

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)



# Make Predictions

predictions = model.predict(X_test)

print("\nRandom Forest Predictions:")
print(predictions)



# Random Forest Accuracy

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nRandom Forest Accuracy:")
print(f"{accuracy:.2f}")



# Actual vs Predicted

print("\nActual vs Predicted:")

for actual, predicted in zip(
    y_test,
    predictions
):
    print(
        f"Actual: {actual} | Predicted: {predicted}"
    )



# Confusion Matrix

cm = confusion_matrix(
    y_test,
    predictions
)

print("\nRandom Forest Confusion Matrix:")
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

for i, prediction in enumerate(
    student_predictions,
    start=1
):

    if prediction == 1:
        result = "Pass"
    else:
        result = "Fail"

    print(
        f"Student {i}: {result}"
    )



# Random Forest n_estimators Comparison

print("\nRandom Forest Comparison:")

for trees in [10, 50, 100, 200]:

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



# Feature Importance

importance = model.feature_importances_
features = X.columns

print("\nFeature Importance:")

for feature, score in zip(
    features,
    importance
):
    print(
        f"{feature}: {score:.3f}"
    )



# Most Important Feature

highest_index = np.argmax(importance)

highest_feature = features[highest_index]
highest_score = importance[highest_index]

print("\nMost Important Feature:")
print(
    f"{highest_feature}: {highest_score:.3f}"
)



# Least Important Feature

least_index = np.argmin(importance)

least_feature = features[least_index]
least_score = importance[least_index]

print("\nLeast Important Feature:")
print(
    f"{least_feature}: {least_score:.3f}"
)



# Sort Feature Importance

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

print("\nSorted Feature Importance:")
print(importance_df)



# Feature Importance Chart

plt.bar(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.title("Random Forest Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()



# Decision Tree Model

decision_tree = DecisionTreeClassifier(
    random_state=42
)

decision_tree.fit(
    X_train,
    y_train
)

decision_predictions = decision_tree.predict(
    X_test
)

decision_accuracy = accuracy_score(
    y_test,
    decision_predictions
)



# Random Forest Model for Comparison

random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest.fit(
    X_train,
    y_train
)

random_predictions = random_forest.predict(
    X_test
)

random_accuracy = accuracy_score(
    y_test,
    random_predictions
)



# Model Comparison

print("\nModel Comparison:")

print(
    f"Decision Tree Accuracy: "
    f"{decision_accuracy:.2f}"
)

print(
    f"Random Forest Accuracy: "
    f"{random_accuracy:.2f}"
)



# Best Model

print("\nBest Model:")

if random_accuracy > decision_accuracy:

    print("Random Forest performed better.")

elif decision_accuracy > random_accuracy:

    print("Decision Tree performed better.")

else:

    print("Both models achieved the same accuracy.")