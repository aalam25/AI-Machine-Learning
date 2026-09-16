import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score
)


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


# Get Prediction Probabilities

y_probability = model.predict_proba(X_test)[:, 1]

print("\nPrediction Probabilities:")
print(y_probability)


# Calculate Precision-Recall Curve

precision, recall, thresholds = precision_recall_curve(
    y_test,
    y_probability
)

print("\nPrecision:")
print(precision)

print("\nRecall:")
print(recall)

print("\nThresholds:")
print(thresholds)


# Calculate Average Precision

average_precision = average_precision_score(
    y_test,
    y_probability
)

print("\nAverage Precision Score:")
print(f"{average_precision:.4f}")


# Create Precision-Recall Curve

plt.figure(figsize=(8, 5))

plt.plot(
    recall,
    precision,
    label=f"Random Forest (AP = {average_precision:.4f})"
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title("Random Forest Precision-Recall Curve")

plt.legend()

plt.tight_layout()

plt.savefig("precision_recall_curve.png")

plt.show()