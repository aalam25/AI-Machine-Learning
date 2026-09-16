import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



# STUDENT PERFORMANCE DATA

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



# CREATE FEATURE MATRIX

X = np.column_stack((
    study_hours,
    attendance,
    assignment_scores,
    exam_scores
))

y = passed



# ORIGINAL FEATURE INFORMATION

print("Original Features:")
print(X)

print("\nFeature Minimums:")
print(X.min(axis=0))

print("\nFeature Maximums:")
print(X.max(axis=0))



# FEATURE SCALING

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nScaled Features:")
print(X_scaled)

print("\nScaled Means:")
print(X_scaled.mean(axis=0))

print("\nScaled Standard Deviations:")
print(X_scaled.std(axis=0))



# TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)



# TRAIN LOGISTIC REGRESSION MODEL

model = LogisticRegression()

model.fit(X_train, y_train)



# MAKE TEST PREDICTIONS

predictions = model.predict(X_test)


print("\nPredictions:")
print(predictions)



# MODEL ACCURACY

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nAccuracy:")
print(f"{accuracy:.2f}")



# NEW STUDENT PREDICTION

new_student = np.array([[
    6,
    90,
    82,
    85
]])

# Scale the new student using the existing scaler
new_student_scaled = scaler.transform(new_student)

prediction = model.predict(
    new_student_scaled
)

probability = model.predict_proba(
    new_student_scaled
)


print("\nNew Student Prediction")
print("----------------------")

if prediction[0] == 1:
    print("Result: Pass")
else:
    print("Result: Fail")

print(
    f"Pass Probability: {probability[0][1]:.2f}"
)



# REUSABLE PREDICTION FUNCTION

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

    student_scaled = scaler.transform(student)

    prediction = model.predict(student_scaled)

    probability = model.predict_proba(student_scaled)

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



# TEST DIFFERENT STUDENTS

predict_student(6, 90, 82, 85)

predict_student(2, 65, 55, 50)

predict_student(8, 98, 95, 95)