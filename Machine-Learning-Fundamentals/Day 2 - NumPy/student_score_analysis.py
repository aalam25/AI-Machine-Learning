import numpy as np

students = ["Tim", "Gim", "Mim", "Kim", "Sim"]

scores = np.array([90, 85, 70, 40, 95])


def get_grade(score):

    if score >= 90:
        return "A"

    elif score >= 80:
        return "B"

    elif score >= 70:
        return "C"

    elif score >= 60:
        return "D"

    else:
        return "F"


print("Student Score Analysis\n")

print("Average Score:", np.mean(scores))
print("Highest Score:", np.max(scores))
print("Lowest Score:", np.min(scores))
print("Total Students:", len(students))

print("\nStudent Scores:")

for student, score in zip(students, scores):

    grade = get_grade(score)

    print(f"{student}: {score} ({grade})")