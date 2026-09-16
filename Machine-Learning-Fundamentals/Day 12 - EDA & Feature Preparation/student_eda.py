import pandas as pd


# Load Dataset

df = pd.read_csv("student_data.csv")



# Preview Dataset

print("First 5 Rows:")
print(df.head())



# Individual Correlations

correlation1 = df["Study_Hours"].corr(df["Exam_Score"])

print("\nStudy Hours vs Exam Score:")
print(correlation1)


correlation2 = df["Attendance"].corr(df["Exam_Score"])

print("\nAttendance vs Exam Score:")
print(correlation2)


correlation3 = df["Assignment_Score"].corr(df["Exam_Score"])

print("\nAssignment Score vs Exam Score:")
print(correlation3)


correlation4 = df["Midterm_Score"].corr(df["Exam_Score"])

print("\nMidterm Score vs Exam Score:")
print(correlation4)



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



# Group Analysis

print("\nAverage Study Hours by Result:")
print(
    df.groupby("Final_Result")["Study_Hours"].mean()
)


print("\nAverage Attendance by Result:")
print(
    df.groupby("Final_Result")["Attendance"].mean()
)


print("\nAverage Assignment Score by Result:")
print(
    df.groupby("Final_Result")["Assignment_Score"].mean()
)


print("\nAverage Midterm Score by Result:")
print(
    df.groupby("Final_Result")["Midterm_Score"].mean()
)


print("\nAverage Exam Score by Result:")
print(
    df.groupby("Final_Result")["Exam_Score"].mean()
)



# Create Features (X)

X = df[
    [
        "Study_Hours",
        "Attendance",
        "Assignment_Score",
        "Midterm_Score",
        "Exam_Score"
    ]
]


print("\nFeatures (X):")
print(X)



# Create Target (y)

y = df["Final_Result"].map({
    "Pass": 1,
    "Fail": 0
})


print("\nEncoded Target (y):")
print(y)



# Check Shapes

print("\nX Shape:")
print(X.shape)


print("\ny Shape:")
print(y.shape)


# Feature Averages

print("\nDataset Summary:")
print(
    df[
        [
            "Study_Hours",
            "Attendance",
            "Assignment_Score",
            "Midterm_Score",
            "Exam_Score"
        ]
    ].mean()
)