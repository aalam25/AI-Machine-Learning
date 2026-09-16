import pandas as pd

students = {
    "Name": ["Tim", "Gim", "Mim", "Kim", "Sim"],
    "Math": [90, 70, 60, 40, 95],
    "Science": [85, 80, 65, 45, 90],
    "English": [88, 75, 70, 50, 95]
}

df = pd.DataFrame(students)

print(df)

print("\nFirst rows:")
print(df.head())

print("\nData information:")
print(df.info())

print("\nStatistics:")
print(df.describe())

print("\nMath scores:")
print(df["Math"])

print("\nAverage Math score:")
print(df["Math"].mean())

print("\nStudents with Math score 80 or higher:")
print(df[df["Math"] >= 80])

print("\nStudents with Science below 60:")
print(df[df["Science"] < 60])

print("\nStudent with English 90 or higher:")
print(df[df["English"] >= 90])

print("\nStudents with Math 70 or greater:")
print(df[df["Math"] >= 70])

df["Average"] = df[["Math", "Science", "English"]].mean(axis=1)
df["Average"] = df["Average"].round(2)
print("\nStudent averages:")
print(df)


top_student = df.loc[df["Average"].idxmax()]

print("\nTop Performing Student:")
print(top_student)


print("\nStudents Needing Improvement:")

print(df[df["Average"] < 60])