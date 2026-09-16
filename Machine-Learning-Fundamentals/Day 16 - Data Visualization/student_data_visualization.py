import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# Load Dataset

df = pd.read_csv("student_data.csv")

print("First 5 Students:")
print(df.head())



# Scatter Plot: Study Hours vs Exam Score

plt.scatter(
    df["Study_Hours"],
    df["Exam_Score"]
)

plt.title("Study Hours vs Exam Score")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")

plt.show()



# Scatter Plot: Attendance vs Exam Score

plt.scatter(
    df["Attendance"],
    df["Exam_Score"]
)

plt.title("Attendance vs Exam Score")
plt.xlabel("Attendance")
plt.ylabel("Exam Score")

plt.show()



# Scatter Plot: Assignment Score vs Exam Score

plt.scatter(
    df["Assignment_Score"],
    df["Exam_Score"]
)

plt.title("Assignment Score vs Exam Score")
plt.xlabel("Assignment Score")
plt.ylabel("Exam Score")

plt.show()



# Scatter Plot: Midterm Score vs Exam Score

plt.scatter(
    df["Midterm_Score"],
    df["Exam_Score"]
)

plt.title("Midterm Score vs Exam Score")
plt.xlabel("Midterm Score")
plt.ylabel("Exam Score")

plt.show()



# Histogram: Exam Score Distribution

plt.hist(
    df["Exam_Score"],
    bins=5
)

plt.title("Exam Score Distribution")
plt.xlabel("Exam Score")
plt.ylabel("Number of Students")

plt.show()



# Histogram: Study Hours Distribution

plt.hist(
    df["Study_Hours"],
    bins=5
)

plt.title("Study Hours Distribution")
plt.xlabel("Study Hours")
plt.ylabel("Number of Students")

plt.show()



# Box Plot: Exam Scores

plt.boxplot(
    df["Exam_Score"]
)

plt.title("Exam Score Box Plot")
plt.ylabel("Exam Score")

plt.show()



# Box Plot: Exam Scores by Final Result

pass_scores = df[
    df["Final_Result"] == "Pass"
]["Exam_Score"]

fail_scores = df[
    df["Final_Result"] == "Fail"
]["Exam_Score"]

plt.boxplot(
    [fail_scores, pass_scores],
    tick_labels=["Fail", "Pass"]
)

plt.title("Exam Scores by Final Result")
plt.xlabel("Final Result")
plt.ylabel("Exam Score")

plt.show()



# Box Plot: Study Hours by Final Result

pass_hours = df[
    df["Final_Result"] == "Pass"
]["Study_Hours"]

fail_hours = df[
    df["Final_Result"] == "Fail"
]["Study_Hours"]

plt.boxplot(
    [fail_hours, pass_hours],
    tick_labels=["Fail", "Pass"]
)

plt.title("Study Hours by Final Result")
plt.xlabel("Final Result")
plt.ylabel("Study Hours")

plt.show()



# Correlation Matrix

correlation = df[
    [
        "Study_Hours",
        "Attendance",
        "Assignment_Score",
        "Midterm_Score",
        "Exam_Score"
    ]
].corr()

print("\nCorrelation Matrix:")
print(correlation)



# Correlation Heatmap

sns.heatmap(
    correlation,
    annot=True
)

plt.title("Student Performance Correlation")

plt.show()