import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score


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


# Calculate ROC Curve Values

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

print("\nFalse Positive Rate:")
print(fpr)

print("\nTrue Positive Rate:")
print(tpr)

print("\nThresholds:")
print(thresholds)


# Calculate AUC Score

auc_score = roc_auc_score(
    y_test,
    y_probability
)

print("\nAUC Score:")
print(f"{auc_score:.4f}")


# Task 8 - Create ROC Curve

plt.figure(figsize=(8, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Random Forest (AUC = {auc_score:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("Random Forest ROC Curve")

plt.legend()

plt.tight_layout()

plt.savefig("roc_curve.png")

plt.show()