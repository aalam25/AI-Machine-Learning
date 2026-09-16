import pandas as pd


# Load student data

df = pd.read_csv("student_data.csv")


# Display original data

print("Original Student Data:")
print(df)


# Check for missing values

print("\nMissing Values:")
print(df.isnull().sum())


# Fill missing English scores

english_average = df["English"].mean()

df["English"] = df["English"].fillna(english_average)


# Calculate student averages

df["Average"] = df[["Math", "Science", "English"]].mean(axis=1)

df["Average"] = df["Average"].round(2)


# Add performance category

def get_category(average):

    if average >= 90:
        return "Excellent"

    elif average >= 75:
        return "Good"

    elif average >= 50:
        return "Average"

    else:
        return "Needs Improvement"


df["Category"] = df["Average"].apply(get_category)


# Display cleaned data

print("\nCleaned Student Data:")
print(df)


# Find students with low attendance

print("\nStudents with Low Attendance:")

low_attendance = df[df["Attendance"] < 75]

print(low_attendance)


# Find top-performing student

top_student = df.loc[df["Average"].idxmax()]

print("\nTop Performing Student:")
print(top_student)


#Average attendance

average_attendance = df["Attendance"].mean()

print(
    f"\nAverage Attendance: {average_attendance:.2f}%"
)