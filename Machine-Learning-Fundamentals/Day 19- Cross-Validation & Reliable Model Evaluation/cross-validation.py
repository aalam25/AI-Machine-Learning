import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier


# Load Dataset

df = pd.read_csv("student_data.csv")


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


# Split Data

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


# Task 4 and Task 5
# 5-Fold Cross-Validation

scores = cross_val_score(
    basic_model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

average_score = scores.mean()

print("\n5-Fold Cross-Validation:")
print("CV Scores:", scores)
print("Average Cross-Validation Accuracy:", average_score)
print("Highest CV Accuracy:", scores.max())
print("Lowest CV Accuracy:", scores.min())


# Task 8
# Compare Different Numbers of Folds

# 3-Fold Cross-Validation

scores2 = cross_val_score(
    basic_model,
    X,
    y,
    cv=3,
    scoring="accuracy"
)

average_score2 = scores2.mean()

print("\n3-Fold Cross-Validation:")
print("CV Scores:", scores2)
print("Average Cross-Validation Accuracy:", average_score2)


# 10-Fold Cross-Validation

scores3 = cross_val_score(
    basic_model,
    X,
    y,
    cv=10,
    scoring="accuracy"
)

average_score3 = scores3.mean()

print("\n10-Fold Cross-Validation:")
print("CV Scores:", scores3)
print("Average Cross-Validation Accuracy:", average_score3)


# Task 9
# Conclusion

print("\nConclusion:")
print(
    "Cross-validation provides a more reliable estimate of model performance "
    "by evaluating the model on multiple train/test splits."
)


# Optional Challenge
# Decision Tree vs Random Forest


# Decision Tree

decision_tree = DecisionTreeClassifier(
    random_state=42
)

dt_scores = cross_val_score(
    decision_tree,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

dt_average = dt_scores.mean()

print("\nDecision Tree:")
print("CV Scores:", dt_scores)
print("Average CV Accuracy:", dt_average)


# Random Forest

random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_scores = cross_val_score(
    random_forest,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

rf_average = rf_scores.mean()

print("\nRandom Forest:")
print("CV Scores:", rf_scores)
print("Average CV Accuracy:", rf_average)


# Compare Models

print("\nModel Comparison:")
print("----------------")

print(
    f"Decision Tree Average Accuracy: "
    f"{dt_average:.2f}"
)

print(
    f"Random Forest Average Accuracy: "
    f"{rf_average:.2f}"
)

if rf_average > dt_average:
    print("\nRandom Forest performed better.")
elif dt_average > rf_average:
    print("\nDecision Tree performed better.")
else:
    print("\nBoth models performed equally.")