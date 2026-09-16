import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# ============================================
# FIRST MACHINE LEARNING MODEL
# Linear Regression: Study Hours → Exam Score
# ============================================

# Dataset

study_hours = np.array([1, 2, 3, 4, 5, 6, 7, 8])
exam_scores = np.array([50, 55, 60, 65, 70, 78, 85, 92])


# ============================================
# DISPLAY DATA
# ============================================

print("=" * 45)
print("STUDENT EXAM SCORE PREDICTION")
print("=" * 45)

print("\nStudy Hours:")
print(study_hours)

print("\nExam Scores:")
print(exam_scores)


# ============================================
# PREPARE FEATURES AND TARGET
# ============================================

X = study_hours.reshape(-1, 1)
y = exam_scores


# ============================================
# SPLIT DATA INTO TRAINING AND TESTING
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================
# CREATE AND TRAIN MODEL
# ============================================

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel training completed.")


# ============================================
# MAKE PREDICTIONS
# ============================================

predictions = model.predict(X_test)

print("\nPredictions:")

for actual, predicted in zip(y_test, predictions):
    print(
        f"Actual: {actual:.2f} | "
        f"Predicted: {predicted:.2f}"
    )


# ============================================
# MODEL EVALUATION
# ============================================

mae = mean_absolute_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)

print("\n" + "=" * 45)
print("MODEL EVALUATION")
print("=" * 45)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")


# ============================================
# MODEL PARAMETERS
# ============================================

print("\n" + "=" * 45)
print("MODEL PARAMETERS")
print("=" * 45)

print(f"Coefficient: {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")


# ============================================
# PREDICT A NEW STUDENT'S SCORE
# ============================================

new_student = np.array([[10]])

predicted_score = model.predict(new_student)

print("\n" + "=" * 45)
print("NEW STUDENT PREDICTION")
print("=" * 45)

print(
    f"Study Hours: 10"
)

print(
    f"Predicted Exam Score: {predicted_score[0]:.2f}"
)


# ============================================
# END
# ============================================

print("\n" + "=" * 45)
print("Analysis Complete")
print("=" * 45)