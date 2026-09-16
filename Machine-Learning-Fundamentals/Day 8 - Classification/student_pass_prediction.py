import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# ============================================
# STUDENT PASS/FAIL PREDICTION
# Logistic Regression Classification Model
# ============================================

# Dataset

study_hours = np.array([
    1, 2, 2, 3, 3, 4, 4, 5,
    5, 6, 6, 7, 7, 8, 8, 9
])

exam_scores = np.array([
    45, 50, 55, 58, 62, 65, 68, 70,
    73, 76, 78, 82, 85, 88, 92, 95
])

passed = np.array([
    0, 0, 0, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1
])


# ============================================
# PREPARE FEATURES AND TARGET
# ============================================

X = exam_scores.reshape(-1, 1)
y = passed


# ============================================
# SPLIT DATA
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)


# ============================================
# CREATE AND TRAIN MODEL
# ============================================

model = LogisticRegression()

model.fit(X_train, y_train)

print("Model training completed.")


# ============================================
# MAKE TEST PREDICTIONS
# ============================================

predictions = model.predict(X_test)

print("\nPredictions:")
print(predictions)


# ============================================
# PREDICTION PROBABILITIES
# ============================================

probabilities = model.predict_proba(X_test)

print("\nPrediction Probabilities:")
print(probabilities)


# ============================================
# MODEL EVALUATION
# ============================================

accuracy = accuracy_score(
    y_test,
    predictions
)

cm = confusion_matrix(
    y_test,
    predictions
)

print("\nModel Evaluation")
print("-" * 30)

print(f"Accuracy: {accuracy:.2f}")

print("\nConfusion Matrix:")
print(cm)


# ============================================
# PREDICT A NEW STUDENT
# ============================================

new_student = np.array([[72]])

prediction = model.predict(new_student)

probability = model.predict_proba(new_student)


print("\nPrediction for Score 72:")
print("-" * 30)

if prediction[0] == 1:
    print("Result: Pass")
else:
    print("Result: Fail")

print(f"Pass Probability: {probability[0][1]:.2f}")


# ============================================
# REUSABLE PREDICTION FUNCTION
# ============================================

def predict_result(score):

    student = np.array([[score]])

    prediction = model.predict(student)

    probability = model.predict_proba(student)

    if prediction[0] == 1:
        result = "Pass"
    else:
        result = "Fail"

    print(f"\nScore: {score}")
    print(f"Prediction: {result}")
    print(f"Pass Probability: {probability[0][1]:.2f}")


# Test the function

predict_result(45)
predict_result(65)
predict_result(85)