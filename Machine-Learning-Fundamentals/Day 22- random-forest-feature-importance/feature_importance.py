import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt


# Task 3 - Load Dataset

df = pd.read_csv("student_data.csv")

print("\nDataset:")
print(df)


# Task 4 - Select Features

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


# Task 5 - Split Dataset

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


# Task 6 - Create Random Forest Model

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

print("\nRandom Forest Model Trained Successfully.")


# Task 7 - Get Feature Importance

importance = model.feature_importances_

print("\nFeature Importance:")
print(importance)


# Task 8 - Create Feature Importance DataFrame

feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance Ranking:")
print(feature_importance_df)


# Task 9 - Identify Most Important Feature

most_important_feature = feature_importance_df.iloc[0]

print("\nMost Important Feature:")
print(most_important_feature["Feature"])

print("\nImportance Value:")
print(f"{most_important_feature['Importance']:.4f}")


# Task 10 - Create Feature Importance Bar Chart

plt.figure(figsize=(8, 5))

plt.bar(
    feature_importance_df["Feature"],
    feature_importance_df["Importance"]
)

plt.title("Random Forest Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("feature_importance.png")

plt.show()



