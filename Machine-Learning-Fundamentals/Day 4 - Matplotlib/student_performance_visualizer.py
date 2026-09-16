import matplotlib.pyplot as plt
import numpy as np


students = ["Tim", "Gim", "Mim", "Kim", "Sim"]

math = [90, 70, 60, 40, 95]
science = [85, 80, 65, 45, 90]
english = [88, 75, 70, 50, 95]

average = np.mean([math, science, english], axis=0)

print("\nAverage Scores:")

for student, score in zip(students, average):
    print(f"{student}: {score:.2f}")
plt.bar(students, average)

plt.title("Student Average Scores")
plt.xlabel("Students")
plt.ylabel("Average Score")

plt.show()

highest_index = np.argmax(average)

highest_student = students[highest_index]
highest_score = average[highest_index]

print(
    f"Top Student: {highest_student}"
)

print(
    f"Average: {highest_score:.2f}"
)

print("\nStudents Needing Improvement:")

for student, score in zip(students, average):
    if score < 60:
        print(f"{student}: {score:.2f}")