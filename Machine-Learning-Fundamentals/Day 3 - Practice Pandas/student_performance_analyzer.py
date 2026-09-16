import pandas as pd


students = {
    "Name": ["Tim", "Gim", "Mim", "Kim", "Sim"],
    "Math": [90, 70, 60, 40, 95],
    "Science": [85, 80, 65, 45, 90],
    "English": [88, 75, 70, 50, 95]
}


df = pd.DataFrame(students)


# Calculate average score

df["Average"] = df[["Math", "Science", "English"]].mean(axis=1)

df["Average"] = df["Average"].round(2)


# Display report

print("Student Performance Report\n")

print(df)


# Find top student

top_student = df.loc[df["Average"].idxmax()]

print("\nTop Performing Student:")
print(top_student)


# Find students needing improvement

print("\nStudents Needing Improvement:")

print(df[df["Average"] < 60])