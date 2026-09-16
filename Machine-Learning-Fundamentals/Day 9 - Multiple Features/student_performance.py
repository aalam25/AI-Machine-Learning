import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# ============================================
# STUDENT PERFORMANCE DATA
# ============================================

study_hours = np.array([
    2, 3, 4, 5, 6,
    7, 8, 3, 5, 7
])

attendance = np.array([
    70, 75, 80, 85, 90,
    95, 98, 78, 88, 92
])

assignment_scores = np.array([
    60, 65, 70, 75, 80,
    85, 90, 68, 78, 88
])

exam_scores = np.array([
    55, 60, 68, 72, 80,
    88, 95, 62, 75, 90
])

# 0 = Fail, 1 = Pass
passed = np.array([
    0, 0, 1, 1, 1,
    1, 1, 0, 1, 1
])


# ============================================
# PREPARE FEATURES AND TARGET
# ============================================

X = np.column_stack((
    study_hours,
    attendance,
    assignment_scores,
    exam_scores
))

y = passed


# ============================================
# TRAIN / TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================
# CREATE AND TRAIN MODEL
# ============================================

model = LogisticRegression()

model.fit(X_train, y_train)

print("Model training completed.")


# ============================================
# MAKE PREDICTIONS
# ============================================

predictions = model.predict(X_test)

print("\nPredictions:")
print(predictions)


# ============================================
# MODEL ACCURACY
# ============================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nAccuracy:")
print(f"{accuracy:.2f}")


# ============================================
# ACTUAL VS PREDICTED
# ============================================

print("\nActual vs Predicted:")

for actual, predicted in zip(y_test, predictions):
    print(
        f"Actual: {actual} | Predicted: {predicted}"
    )


# ============================================
# CONFUSION MATRIX
# ============================================

cm = confusion_matrix(
    y_test,
    predictions
)

print("\nConfusion Matrix:")
print(cm)


# ============================================
# NEW STUDENT PREDICTION
# ============================================

new_student = np.array([[
    6,      # Study Hours
    90,     # Attendance
    82,     # Assignment Score
    85      # Exam Score
]])

prediction = model.predict(new_student)

probability = model.predict_proba(new_student)

print("\nNew Student Prediction")
print("----------------------")

if prediction[0] == 1:
    print("Result: Pass")
else:
    print("Result: Fail")

print(
    f"Pass Probability: {probability[0][1]:.2f}"
)


# ============================================
# REUSABLE PREDICTION FUNCTION
# ============================================

def predict_student(
    study_hours,
    attendance,
    assignment_score,
    exam_score
):

    student = np.array([[
        study_hours,
        attendance,
        assignment_score,
        exam_score
    ]])

    prediction = model.predict(student)

    probability = model.predict_proba(student)

    if prediction[0] == 1:
        result = "Pass"
    else:
        result = "Fail"

    print("\nStudent Prediction")
    print("------------------")
    print(f"Study Hours: {study_hours}")
    print(f"Attendance: {attendance}%")
    print(f"Assignment Score: {assignment_score}")
    print(f"Exam Score: {exam_score}")
    print(f"Prediction: {result}")
    print(
        f"Pass Probability: {probability[0][1]:.2f}"
    )


# ============================================
# TEST DIFFERENT STUDENTS
# ============================================

predict_student(6, 90, 82, 85)

predict_student(2, 65, 55, 50)

predict_student(8, 98, 95, 95)