import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

import matplotlib.pyplot as plt


# Load Dataset

df = pd.read_csv("student_data.csv")

print("\nDataset:")
print(df)


# Select Features and Target

X = df[
    [
        "Study_Hours",
        "Attendance",
        "Assignment_Score",
        "Exam_Score"
    ]
]

y = df["Final_Result"].map({
    "Pass": 1,
    "Fail": 0
})

print("\nFeatures:")
print(X)

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
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

print("\nRandom Forest Model Trained Successfully.")


# Calculate Traditional Feature Importance

traditional_importance = model.feature_importances_

traditional_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": traditional_importance
})

traditional_df = traditional_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTraditional Feature Importance:")
print(traditional_df)


# Calculate Permutation Importance

permutation = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="accuracy"
)


# Create Permutation Importance DataFrame

permutation_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": permutation.importances_mean
})

permutation_df = permutation_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nPermutation Feature Importance:")
print(permutation_df)


# Create Permutation Importance Visualization

plt.figure(figsize=(8, 5))

plt.bar(
    permutation_df["Feature"],
    permutation_df["Importance"]
)

plt.title("Random Forest Permutation Feature Importance")
plt.xlabel("Features")
plt.ylabel("Mean Importance")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("permutation_importance.png")

plt.show()


# Identify Most Important Feature

most_important = permutation_df.iloc[0]

print("\nMost Important Feature According to Permutation Importance:")
print(most_important["Feature"])

print("\nImportance Value:")
print(f"{most_important['Importance']:.4f}")