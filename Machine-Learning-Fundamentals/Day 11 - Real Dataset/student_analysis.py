import pandas as pd


# Load Dataset
df = pd.read_csv("student_data.csv")


# Basic Dataset Exploration
print("First 5 Students:")
print(df.head())


print("\nDataset Shape:")
print(df.shape)


print("\nColumn Names:")
print(df.columns)


print("\nData Types:")
print(df.dtypes)


print("\nDataset Information:")
df.info()


print("\nStatistical Summary:")
print(df.describe())


# Missing Value Analysis
print("\nMissing Values:")
print(df.isnull().sum())


# Basic Statistics
print("\nAverage Study Hours:")
print(df["Study_Hours"].mean())


print("\nAverage Attendance:")
print(df["Attendance"].mean())


print("\nAverage Exam Score:")
print(df["Exam_Score"].mean())


print("\nHighest Exam Score:")
print(df["Exam_Score"].max())


print("\nLowest Exam Score:")
print(df["Exam_Score"].min())


# Highest Performing Student
print("\nHighest Exam Scorer:")
print(df.loc[df["Exam_Score"].idxmax()])


# Pass/Fail Analysis
print("\nPass/Fail Count:")
print(df["Final_Result"].value_counts())


pass_rate = (df["Final_Result"] == "Pass").mean()

print("\nOverall Pass Rate:")
print(f"{pass_rate:.2%}")


# Passing and Failing Students
print("\nAverage Passing Exam Score:")
print(
    df[df["Final_Result"] == "Pass"]["Exam_Score"].mean()
)


print("\nAverage Failing Exam Score:")
print(
    df[df["Final_Result"] == "Fail"]["Exam_Score"].mean()
)


# Students with High Study Hours and Attendance
print("\nStudents with High Study Hours and Attendance:")
print(
    df[
        (df["Study_Hours"] >= 7) &
        (df["Attendance"] >= 90)
    ]
)